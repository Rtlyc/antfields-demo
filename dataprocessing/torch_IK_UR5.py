# import PyKDL
# import tf_conversions.posemath
from math import *
import numpy as np
import torch

def invTransform(T):
	#n*4*4
	#n*3*3 @ n*3*1
	R = T[:,:3,:3]
	t = T[:,:3,3]
	Rt = torch.transpose(R, 1, 2)

	inverseT = torch.zeros((T.shape[0],4,4), device='cuda')

	inverseT[:,:3,:3] = Rt
	#print(Rt.shape)
	#print(t.shape)
	#print(inverseT[:,:3,3].shape)
	#print((Rt@t.unsqueeze(2)).shape)
	inverseT[:,:3,3] = - (Rt@t.unsqueeze(2)).squeeze()
	inverseT[:,3,3] = 1

	return inverseT

def transformDHParameter(a,d,alpha,theta):
	T = torch.zeros((theta.shape[0],4,4), device='cuda')
	
	cos_theta = torch.cos(theta)
	sin_theta = torch.sin(theta)
	cos_alpha = torch.cos(alpha)
	sin_alpha = torch.sin(alpha)
	
	T[:,0,0] = cos_theta
	T[:,0,1] = -sin_theta*cos_alpha
	T[:,0,2] = sin_theta*sin_alpha
	T[:,0,3] = cos_theta*a

	T[:,1,0] = sin_theta
	T[:,1,1] = cos_theta*cos_alpha
	T[:,1,2] = -cos_theta*sin_alpha
	T[:,1,3] = sin_theta*a

	T[:,2,1] = sin_alpha
	T[:,2,2] = cos_alpha
	T[:,2,3] = d

	T[:,3,3] = 1

	return T

def transformRobotParameter(theta):
	d = [0.089159,0,0,0.10915,0.09465,0.0823]
	a = [0,-0.425,-0.39225,0,0,0]
	alpha = [pi/2,0,0,pi/2,-pi/2,0]
	T = np.eye(4)
	for i in range(6):
		T = T.dot(transformDHParameter(a[i],d[i],alpha[i],theta[i]))
	return T

class torch_IK_UR5:
	def __init__(self, batch_size):
		# Debug mode
		self.debug = False

		# Robot DH parameter
		self.d = torch.tensor([0.1625,0,0,0.1333,0.0997,0.0996],device='cuda').float()
		self.a = torch.tensor([0,-0.425,-0.3922,0,0,0],device='cuda').float()
		#self.d = torch.tensor([0.089159,0,0,0.10915	,0.09465,0.0823],device='cuda').float()
		#self.a = torch.tensor([0,-0.425,-0.39225,0,0,0],device='cuda').float()
		
		self.alpha = torch.tensor([pi/2,0,0,pi/2,-pi/2,0],device='cuda').float()

		# Robot EE orientation offset.
		# Useful if the ee orientation when the all joint = 0 is not
		#     1  0  0
		# R = 0  0 -1
		#     0  1  0
		# ee_offset = current_R_all_joint_0.transpose * R

		# Robot joint limits
		self.limit_max = 2 * pi
		self.limit_min = -2 * pi

		# Robot joint weights
		self.joint_weights = np.array([1,1,1,1,1,1])

		self.d5 = torch.tensor([0,0,-self.d[5],1],device='cuda').float()
		self.d3 = torch.tensor([0,-self.d[3],0,1],device='cuda').float()
		self.zero = torch.tensor([0,0,0,1],device='cuda').float()

		# Robot target transformation
		x = torch.eye(4, device= 'cuda').float()
		x = x.reshape((1, 4, 4))
		self.gd = x.repeat(batch_size, 1, 1)
		#self.gd = np.identity(4)

		# Stopping IK calculation flag
		self.stop_flag = False

		# Robot joint solutions data
		self.theta1 = torch.zeros((batch_size,2),device= 'cuda').float()

		self.flags1 = None

		self.theta5 = torch.zeros((batch_size,2,2),device= 'cuda').float()

		self.flags5 = None

		self.theta6 = torch.zeros((batch_size,2,2),device= 'cuda').float()

		self.theta2 = torch.zeros((batch_size,2,2,2),device= 'cuda').float()
		self.theta3 = torch.zeros((batch_size,2,2,2),device= 'cuda').float()

		self.flags3 = None

		self.theta4 = torch.zeros((batch_size,2,2,2),device= 'cuda').float()

	def enableDebugMode(self, debug = True):
		# This function will enable/disable debug mode
		self.debug = debug

	def setJointLimits(self, limit_min, limit_max):
		# This function is used to set the joint limit for all joint
		self.limit_max = limit_max
		self.limit_min = limit_min

	def setJointWeights(self, weights):
		# This function will assign weights list for each joint
		self.joint_weight = np.array(weights)

	def normalize(self,value):
		# This function will normalize the joint values according to the joint limit parameters
		normalized = value
		#print(normalized)
		where = normalized > self.limit_max
		while len(normalized[where])>0:
			normalized[where] -= 2 * pi
			where = normalized > self.limit_max
		where = normalized < self.limit_min
		while len(normalized[where])>0:
			normalized[where] += 2* pi
			where = normalized < self.limit_min
		#print(normalized)
		return normalized

	def getFlags(self,nominator,denominator):
		# This function is used to check whether the joint value will be valid or not
		if denominator == 0:
			return False
		return abs(nominator/denominator) < 1.01

	def getTheta1(self):
		# This function will solve joint 1
		self.flags1 = np.ones(2)
		#gd: n*4*4
		#p05: n*4*1
		p05 = self.gd@self.d5-self.zero
		#print(self.gd.shape)
		psi = torch.atan2(p05[:,1],p05[:,0])

		L = torch.sqrt(p05[:,0]**2+p05[:,1]**2)

		# gives tolerance if acos just a little bit bigger than 1 to return
		# real result, otherwise the solution will be flagged as invalid
		where = abs(self.d[3]) > L
		L[where] = abs(self.d[3])
		'''
		if abs(self.d[3]) > L:
			if self.debug:
				print('L1 = ', L, ' denominator = ', self.d[3])
			self.flags1[:] = self.getFlags(self.d[3],L) # false if the ratio > 1.001
			L = abs(self.d[3])
		'''
		phi = torch.acos(self.d[3]/L)

		self.theta1[:,0] = self.normalize(psi+phi+pi/2)
		self.theta1[:,1] = self.normalize(psi-phi+pi/2);

		# stop the program early if no solution is possible

		self.stop_flag = not np.any(self.flags1)
		if self.debug:
			print('t1: ', self.theta1)
			print('flags1: ',self.flags1)
	
	def getTheta5(self):
		# This function will solve joint 5
		self.flags5 = np.ones((2,2))

		p06 = self.gd[:,0:3,3]
		for i in range(2):
			p16z = p06[:,0]*torch.sin(self.theta1[:,i])-p06[:,1]*torch.cos(self.theta1[:,i]);
			L = self.d[5]

			where = abs(p16z - self.d[3]) > L
			#print(p16z.shape)
			L = torch.ones((p16z.shape[0]),device='cuda')*L
			L[where] = abs(p16z[where]-self.d[3])
			'''
			if abs(p16z - self.d[3]) > L:
				if self.debug:
					print('L5 = ', L, ' denominator = ', abs(p16z - self.d[3]))
				self.flags5[i,:] = self.getFlags(p16z - self.d[3],self.d[5])
				L = abs(p16z-self.d[3]);
			'''
			theta5i = torch.acos((p16z-self.d[3])/L)
			self.theta5[:,i,0] = theta5i
			self.theta5[:,i,1] = -theta5i

		# stop the program early if no solution is possible
		self.stop_flag = not np.any(self.flags5)
		if self.debug:
			print('t5: ', self.theta5)
			print('flags5: ',self.flags5)

	def getTheta6(self):
		# This function will solve joint 6
		for i in range(2):
			#T1: n*4*4
			T1 = transformDHParameter(self.a[0],self.d[0],self.alpha[0],self.theta1[:,i])
			T61 = invTransform(invTransform(T1)@self.gd)
			for j in range(2):
				where = torch.sin(self.theta5[:,i,j]) != 0
				self.theta6[~where,i,j] = 0
				self.theta6[where,i,j] = torch.atan2(-T61[where,1,2]/torch.sin(self.theta5[where,i,j]),
											  T61[where,0,2]/torch.sin(self.theta5[where,i,j]))
				'''
				if torch.sin(self.theta5[:,i,j]) == 0:
					if self.debug:
						print("Singular case. selected theta 6 = 0")
					self.theta6[:,i,j] = 0
				else:
					self.theta6[:,i,j] = torch.atan2(-T61[:,1,2]/torch.sin(self.theta5[:,i,j]),
											  T61[:,0,2]/torch.sin(self.theta5[:,i,j]))
				'''
		# print 't6: ', self.theta6

	def getTheta23(self):
		# This function will solve joint 2 and 3
		self.flags3 = np.ones ((2,2,2))
		for i in range(2):
			T1 = transformDHParameter(self.a[0],self.d[0],self.alpha[0],self.theta1[:,i])
			T16 = invTransform(T1)@ self.gd
			#print(T16.shape)
			for j in range(2):
				T45 = transformDHParameter(self.a[4],self.d[4],self.alpha[4],self.theta5[:,i,j])
				T56 = transformDHParameter(self.a[5],self.d[5],self.alpha[5],self.theta6[:,i,j])
				T14 = T16@ invTransform(T45@T56)

				P13 = T14 @ self.d3 - self.zero
				#print(P13.shape)
				L = torch.norm(P13,dim=1)**2 - self.a[1]**2 - self.a[2]**2

				where = abs(L / (2*self.a[1]*self.a[2]) ) > 1
				L[where] = torch.sign(L[where]) * 2*self.a[1]*self.a[2]
				'''
				if abs(L / (2*self.a[1]*self.a[2]) ) > 1:
					if self.debug:
						print('L3 = ', L, ' denominator = ', (2*self.a[1]*self.a[2]))
					self.flags3[i,j,:] = self.getFlags(L,2*self.a[1]*self.a[2])
					L = np.sign(L) * 2*self.a[1]*self.a[2]
				'''
				self.theta3[:,i,j,0] = torch.acos(L / (2*self.a[1]*self.a[2]) )
				self.theta2[:,i,j,0] = -torch.atan2(P13[:,1],-P13[:,0]) + torch.asin( self.a[2]*torch.sin(self.theta3[:,i,j,0])/torch.norm(P13,dim=1) )
				self.theta3[:,i,j,1] = -self.theta3[:,i,j,0]
				self.theta2[:,i,j,1] = -torch.atan2(P13[:,1],-P13[:,0]) + torch.asin( self.a[2]*torch.sin(self.theta3[:,i,j,1])/torch.norm(P13,dim=1) )
		if self.debug:
			print('t2: ', self.theta2)
			print('t3: ', self.theta3)
			print('flags3: ',self.flags3)

		# stop the program early if no solution is possible
		self.stop_flag = not np.any(self.flags3)
	
	def getTheta4(self):
		# This function will solve joint 4 value
		for i in range(2):
			T1 = transformDHParameter(self.a[0],self.d[0],self.alpha[0],self.theta1[:,i])
			T16 = invTransform(T1) @ self.gd
			
			for j in range(2):
				T45 = transformDHParameter(self.a[4],self.d[4],self.alpha[4],self.theta5[:,i,j])
				T56 = transformDHParameter(self.a[5],self.d[5],self.alpha[5],self.theta6[:,i,j])
				T14 = T16 @ invTransform(T45@ T56)

				for k in range(2):
					T13 = transformDHParameter(self.a[1],self.d[1],self.alpha[1],self.theta2[:,i,j,k])@ \
						  transformDHParameter(self.a[2],self.d[2],self.alpha[2],self.theta3[:,i,j,k]) 
					T34 = invTransform(T13) @ T14
					self.theta4[:,i,j,k] = torch.atan2(T34[:,1,0],T34[:,0,0])
		if self.debug:
			print('t4: ', self.theta4)

	def countValidSolution(self):
		# This function will count the number of available valid solutions
		number_of_solution = 0
		for i in range(2):
			for j in range(2):
				for k in range(2):
					if self.flags1[i] and self.flags3[i,j,k] and self.flags5[i,j]:
						number_of_solution += 1
		return number_of_solution

	def getSolution(self):
		# This function will call all function to get all of the joint solutions
		for i in range(4):
			if i == 0:
				#print('1')
				self.getTheta1()
			elif i == 1:
				#print('5')
				self.getTheta5()
			elif i == 2:
				#print('6')
				self.getTheta6()
				#print('23')
				self.getTheta23()
			elif i == 3:
				#print('4')
				self.getTheta4()

			# This will stop the solving the IK when there is no valid solution from previous joint calculation
			if self.stop_flag:
				return

	def solveIK(self,forward_kinematics):
		#tensor n*4*4
		self.gd = forward_kinematics
		#print(self.gd.shape)
		if self.debug:
			print('Input to IK:\n', self.gd)
		self.getSolution()
		'''
		number_of_solution = self.countValidSolution()
		if self.stop_flag or number_of_solution < 1:
			if self.debug:
				print('No solution')
			return None
		'''
		Q = torch.zeros((forward_kinematics.shape[0],8,6),device='cuda')
		index = 0
		for i in range(2):
			for j in range(2):
				for k in range(2):
					'''
					if not (self.flags1[i] and self.flags3[i,j,k] and self.flags5[i,j]):
						# skip invalid solution
						continue
					'''
					#print(self.theta1[i].shape)
					Q[:,index,0] = self.normalize(self.theta1[:,i])
					Q[:,index,1] = self.normalize(self.theta2[:,i,j,k])
					Q[:,index,2] = self.normalize(self.theta3[:,i,j,k])
					Q[:,index,3] = self.normalize(self.theta4[:,i,j,k])
					Q[:,index,4] = self.normalize(self.theta5[:,i,j])
					Q[:,index,5] = self.normalize(self.theta6[:,i,j])
					index += 1

		return Q
	
	def solveIK_one(self,forward_kinematics):
		#tensor n*4*4
		self.gd = forward_kinematics
		#print(self.gd.shape)
		if self.debug:
			print('Input to IK:\n', self.gd)
		self.getSolution()
		'''
		number_of_solution = self.countValidSolution()
		if self.stop_flag or number_of_solution < 1:
			if self.debug:
				print('No solution')
			return None
		'''
		Q = torch.zeros((forward_kinematics.shape[0],1,6),device='cuda')
		index = 0
		for i in range(2):
			for j in range(2):
				for k in range(2):
					'''
					if not (self.flags1[i] and self.flags3[i,j,k] and self.flags5[i,j]):
						# skip invalid solution
						continue
					'''
					#print(self.theta1[i].shape)
					if index==7:
						Q[:,0,0] = self.normalize(self.theta1[:,i])
						Q[:,0,1] = self.normalize(self.theta2[:,i,j,k])
						Q[:,0,2] = self.normalize(self.theta3[:,i,j,k])
						Q[:,0,3] = self.normalize(self.theta4[:,i,j,k])
						Q[:,0,4] = self.normalize(self.theta5[:,i,j])
						Q[:,0,5] = self.normalize(self.theta6[:,i,j])
					index += 1

		return Q
