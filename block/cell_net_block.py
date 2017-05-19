from NNL.packageList import *
import torch.nn.functional as F
import collections
import math
import re


class conv_block(nn.Module):
	def __init__(self,before,after):
		super().__init__()
		self.conv = nn.Conv2d(before,after,(3,3),padding=1)
	def forward(self,x):
		return self.conv(x)

class maxPool_block(nn.Module):
	def __init__(self,size,stride):
		super().__init__()
		self.maxpool = nn.MaxPool2d((size,size),stride=stride)
	def forward(self,x):
		return self.maxpool(x)

class linear_block(nn.Module):
	def __init__(self,before,after):
		super().__init__()
		self.linear = nn.Linear(before,after)
	def forward(self,x):
		return self.linear(x)

class dropout_block(nn.Module):
	def __init__(self):
		super().__init__()
		self.drop = nn.Dropout2d()
	def forward(self,x):
		return self.drop(x)

class BatchNormalize_block(nn.Module):
	def __init__(self,num):
		super().__init__()
		self.normalize = nn.BatchNorm2d(num)
	def forward(self,x):
		return self.normalize(x)

##################################################################
#activation layers
##################################################################
class ReLU_block(nn.Module):
	def __init__(self):
		super().__init__()
		self.ReLU = nn.ReLU()
	def forward(self,x):
		return self.ReLU(x)

class Sigmoid_block(nn.Module):
	def __init__(self):
		super().__init__()
		self.Sigmoid = nn.Sigmoid()
	def forward(self,x):
		return self.Sigmoid(x)

class Tanh_block(nn.Module):
	def __init__(self):
		super().__init__()
		self.Tanh = nn.Tanh()
	def forward(self,x):
		return self.Tanh(x)



class ResnetBlock(nn.Module):
	def __init__(self,structure):
		super().__init__()
		self.Layer_struct = structure
		self.forward_layer = self.layer()
	def forward(self,x):
		residual = x
		x = self.forward_layer(x)
		x = x+residual
		return x

	def layer(self):
		layers=[]
		for line in self.Layer_struct:
			_type = line[0]
			if(_type=='conv'):
				_before, _after = int(line[1]), int(line[2])
				layers.append(conv_block(_before,_after))
			elif(_type=='drop'):
				layers.append(dropout_block())
			elif(_type=='norm'):
				_before = int(line[1])
				layers.append(BatchNormalize_block(_before))
			elif(_type=='maxpool'):
				_before_size, _after_size = int(line[1]), int(line[2])
				if(abs(_after_size-_before_size)==1):
					size = 2
					stride = 1
				elif(math.floor(_before_size/_after_size)==math.ceil(_before_size/_after_size)):
					size = math.floor(_before_size/_after_size)
					stride = math.floor(_before_size/_after_size)
				layers.append(maxPool_block(size,stride))
			elif(_type=='linear'):
				_before, _after = int(line[1]), int(line[2])
				layers.append(linear_block(_before,_after))
			elif(_type=='relu'):
				layers.append(ReLU_block())
			elif(_type=='sigmoid'):
				layers.append(Sigmoid_block())
		return nn.Sequential(*layers)



class BasicBlock(nn.Module):
	def __init__(self,structure):
		super().__init__()
		self.Layer_struct = structure

		self.forward_layer = self.layer()

	def forward(self,x):
		x = self.forward_layer(x)

		return x

	def layer(self):
		layers=[]
		for line in self.Layer_struct:
			_type = line[0]	
			if(_type=='conv'):
				_before, _after = int(line[1]), int(line[2])
				layers.append(conv_block(_before,_after))
			elif(_type=='drop'):
				layers.append(dropout_block())
			elif(_type=='norm'):
				_before = int(line[1])
				layers.append(BatchNormalize_block(_before))
			elif(_type=='maxpool'):
				_before_size, _after_size = int(line[1]), int(line[2])
				if(abs(_after_size-_before_size)==1):
					size = 2
					stride = 1
				elif(math.floor(_before_size/_after_size)==math.ceil(_before_size/_after_size)):
					size = math.floor(_before_size/_after_size)
					stride = math.floor(_before_size/_after_size)
				layers.append(maxPool_block(size,stride))
			elif(_type=='linear'):
				_before, _after = int(line[1]), int(line[2])
				layers.append(linear_block(_before,_after))
			elif(_type=='relu'):
				layers.append(ReLU_block())
			elif(_type=='sigmoid'):
				layers.append(Sigmoid_block())
		return nn.Sequential(*layers)

class Net(nn.Module):
	def __init__(self, profile_string):
		super().__init__()
		self.profile_string = profile_string
		self.profiles = self.analize_profile()
		self.layers = []
		# if(len(self.profiles)!=2):
		# 	raise ValueError('not two profiles')
		self.Normallayer, self.Linearlayer = self._make_block(self.profiles)
	def forward(self,x):
		x = self.preprocess(x)

		x = self.Normallayer(x)

		x = self.afterprocess(x)

		x =self.Linearlayer(x)

		return x

	def preprocess(self,x):
		x = x[:,0,:,:]
		x = torch.unsqueeze(x,1)
		return x

	def afterprocess(self,x):
		x = torch.squeeze(x)
		# x = x.view(-1,32)
		return x

	def _make_layer(self,profile):
		if(profile[0][0]=='linear'):
			linear = 1
		else:
			linear = 0

		if(profile[0]=="res.start")and(profile[-1]=="res.end"):
			layer = nn.Sequential(*[ResnetBlock(profile)])
		else:
			layer = nn.Sequential(*[BasicBlock(profile)])
		return layer, linear

	def _make_block(self,profiles):
		normal_block = []
		linear_block = []
		for line in profiles:
			layer,linear = self._make_layer(line)
			if(linear==0):
				normal_block.append(layer)
			else:
				linear_block.append(layer)
		return nn.Sequential(*normal_block), nn.Sequential(*linear_block)
		#next, enable linear,nolinear,linear structure, currently only support nolinear,linear bisection structure

	def analize_profile(self):
		Dict = {'*':'conv','->':'drop','+.':'norm','--':'maxpool','|':'linear','R':'relu','S':'sigmoid','res.start':'res.start','res.end':'res.end'}
		List = self.profile_string.split()
		self.Layer_struct = []
		for i,item in enumerate(List):
		    if(i%2==1):
		        self.Layer_struct.append([List[i],List[i-1],List[i+1]])

		period = []
		temp = []
		linear_start = 0
		for _type, _before, _after in self.Layer_struct:
			if(_type!="--"):
				if(re.search('\(',_before)!=None):
					_before = _before.split("(")[0]
				if(re.search('\(',_after)!=None):
					num = _after.split("(")[0]
					content = _after.split("(")[1].split(")")[0].split(",")
					#         print(content)
					temp.append([_type,_before,num])
					for item in content:
						if(item=="{"):
							period.append(temp)
							temp=[['res.start','start']]
						elif(item=="}"):
							temp.append(['res.end','end'])
							period.append(temp)
							temp = []
						else:
							temp.append([item,num])
				else:
					temp.append([_type,_before,_after])
			else:
				if(re.search('\(',_before)!=None):
					_before = _before.split(")")[-2].split("(")[1]
				if(re.search('\(',_after)!=None):
					_after = _after.split(")")[-2].split("(")[1]
				temp.append([_type,_before,_after])
		period.append(temp)
		profiles = []
		for profile in period:
			temp = []
			temp1 = []
			linear_start = 0
			for item in profile:
				item[0] = Dict[item[0]]
				if((item[0]!='linear')and(linear_start==0)):
					temp.append(item)
				else:
					linear_start = 1
					temp1.append(item)
			if(len(temp)!=0):
				profiles.append(temp)
			if(len(temp1)!=0):
				profiles.append(temp1)
		for line in profiles:
			print(line)
		return profiles