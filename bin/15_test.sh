#!/bin/bash

# # 测试 Mapper
# # Linux
# cat input/15.BigData_MapReduce/inputFile.txt | python src/python/15.BigData_MapReduce/mrMeanMapper.py
# # # Window
# # python src/python/15.BigData_MapReduce/mrMeanMapper.py < input/15.BigData_MapReduce/inputFile.txt

# # 测试 Reducer
# # Linux
# cat input/15.BigData_MapReduce/inputFile.txt | python src/python/15.BigData_MapReduce/mrMeanMapper.py | python src/python/15.BigData_MapReduce/mrMeanReducer.py
# # # Window
# # python src/python/15.BigData_MapReduce/mrMeanMapper.py < input/15.BigData_MapReduce/inputFile.txt | python src/python/15.BigData_MapReduce/mrMeanReducer.py

# 测试 mrjob的案例
# 先测试一下mapper方法
# python src/python/15.BigData_MapReduce/mrMean.py --mapper < input/15.BigData_MapReduce/inputFile.txt
# 运行整个程序，移除 --mapper 就行
python src/python/15.BigData_MapReduce/mrMean.py < input/15.BigData_MapReduce/inputFile.txt
