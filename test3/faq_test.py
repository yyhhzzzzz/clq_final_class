from data_process import (
    read_corpus,
    get_question_list,
    input_question_process,
    ques_idx_cosine_sim,
)

# 步骤1: 获取问题列表和答案列表并对问题列表进行预处理
# 步骤1.1: 获取问题列表和答案列表
questions = read_corpus('./test3/data/questions.txt')
answers = read_corpus('./test3/data/answers.txt')

# 步骤1.2: 对问题列表进行预处理
questions_list = get_question_list(questions)

# 步骤2: 进行FAQ问答系统测试
print('欢迎您使用FAQ问答系统...')
print("="*50)
while True:
    # 步骤2.1: 输入问题
    input_ques = input('请输入您需要了解的新冠病毒问题(输入q退出系统):\n')
    if input_ques == 'q':
        print('谢谢您的关注！')
        break
    else:
        # 步骤2.2: 处理输入的问题
        ques_process = input_question_process(questions_list, input_ques)

        # 步骤2.3: 获取与输入问题有最大相似度值的问题索引并给出对应的答案
        print('正在FAQ库中寻找答案，请稍等...')
        print("-"*50)
        answer_idx = ques_idx_cosine_sim(ques_process[-1], ques_process[0:-1])
        if answer_idx is not None:
            print('亲，我们给您找到的答案如下:\n', answers[answer_idx])
            print('FAQ库中相似的问题: ', questions[answer_idx])
            print("="*50)