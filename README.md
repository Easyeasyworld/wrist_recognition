# 数据集说明
所有数据集相关信息：https://ieee-dataport.org/documents/gesture-recognition-and-biometrics-electromyography-grabmyo-dataset
简要说明：43个subject的手腕和前臂肌电信号。17个动作（17号动作是休息状态），7个trial。  
          18-23号通道，26-31号通道是手腕肌电单端通道，需要对应相减得到差分信号后再进行分类。
		  eg. 18-26得到第一个差分信号，19-27得到第二个差分通道，以此类推...
	
