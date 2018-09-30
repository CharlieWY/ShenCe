import pandas as pd
data=pd.read_csv('all_docs.txt',sep='\001',header=None)  #读入并处理数据
data.columns=['id','title','doc']

train=pd.read_csv('train_docs_keywords.txt',sep='\t',header=None)
train.columns=['id','label']
train_id_list=list(train['id'].unique())

#构造训练集
train_title_doc=data[data['id'].isin(train_id_list)]
#构造测试集
test_title_doc=data[~data['id'].isin(train_id_list)]
#使用merge方法采用inner的方法连接这两个dataframe
train_title_doc=pd.merge(train_title_doc,train,on=['id'],how='inner')


import jieba
jieba.load_userdict('extra.txt') #载入自定义词典
import re
import jieba.analyse
import numpy as np

# 去除文章的数字，数字没有意义，单纯一个数字，不能达到对文章内容的区分
train_title_doc['title_cut'] = train_title_doc['title'].apply(lambda x:''.join(filter(lambda ch: ch not in ' \t1234567890', x)))

# 策略 extract_tags 直接利用jieba的提取主题词的工具
train_title_doc['title_cut'] = train_title_doc['title_cut'].apply(lambda x:','.join(jieba.analyse.extract_tags(x,topK = 5)))
# 第二规则 提取 《》 通过分析发现，凡是书名号的东西都会被用来作为主题词
train_title_doc['title_regex'] = train_title_doc['title'].apply(lambda x:','.join(re.findall(r"《(.+?)》",x)))


# 利用策略 + 规则 查看训练集的准确率
train_offline_result = train_title_doc[['id','label','title_cut','title_regex']]

# 验证我这个规则能够达到的分数 记得 * 0.5
count = 0
for i in train_offline_result.values:
    result = str(i[1]).split(',')
    title_cut = str(i[2]).split(',')
    title_regex = str(i[3]).split(',')
    if title_regex[0] == '':
        tmp_result = title_cut
    else:
        tmp_result = title_regex + title_cut

    count = count + len(set(result[:2])&set(tmp_result[:2]))
    print(count)

# 策略 extract_tags
test_title_doc['title_cut'] = test_title_doc['title'].apply(lambda x:''.join(filter(lambda ch: ch not in ' \t1234567890', str(x))))

test_title_doc['title_cut'] = test_title_doc['title_cut'].apply(lambda x:','.join(jieba.analyse.extract_tags(str(x),topK = 5)))
# 第二规则 提取 《》
test_title_doc['title_regex'] = test_title_doc['title'].apply(lambda x:','.join(re.findall(r"《(.+?)》",str(x))))

# 利用策略 + 规则 查看训练集的准确率
test_offline_result = test_title_doc[['id','id','title_cut','title_regex']]

label1 = []
label2 = []

for i in test_offline_result.values:
    result = str(i[1]).split(',')
    title_cut = str(i[2]).split(',')
    title_regex = str(i[3]).split(',')
    if title_regex[0] == '':
        tmp_result = title_cut
    else:
        tmp_result = title_regex + title_cut

    if len(tmp_result) > 1:
        label1.append(tmp_result[0])
        label2.append(tmp_result[1])
    elif len(tmp_result) == 1:
        label1.append(tmp_result[0])
        label2.append(tmp_result[0])
    else:
        label1.append('')
        label2.append('')

result = pd.DataFrame()

id = test_title_doc['id'].unique()

result['id'] = list(id)
result['label1'] = label1
result['label1'] = result['label1'].replace(',','nan')
result['label2'] = label2
result['label2'] = result['label2'].replace(',','nan')

result.to_csv('result.csv',index=None,encoding='utf_8_sig')