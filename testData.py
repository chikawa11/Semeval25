import datasets

data_path = "/disk/disk_20T/share/semeval25/train"
forget_data = datasets.load_dataset(path=data_path,data_files="forget.jsonl")['train'][0]
retain_data = datasets.load_dataset(path=data_path,data_files="retain.jsonl")['train'][0]
forget_data_task1 = []
forget_data_task2 = []
forget_data_task3 = []
cc = 0
for id, task in forget_data['task'].items():
    if task == "Task3":
        cc+=1
print(cc)
for id, task in retain_data['task'].items():
    if task == "Task3":
        cc+=1
print(cc)


    # if task == "Task1":
    #     forget_data_task1.append({'question': forget_data['input'][id], 'answer': forget_data['output'][id]})
    # elif task == "Task2":
    #     forget_data_task2.append({'question': forget_data['input'][id], 'answer': forget_data['output'][id]})
    # elif task == "Task3":
    #     forget_data_task3.append({'question': forget_data['input'][id], 'answer': forget_data['output'][id]})
# cc = 0
# for id, out in forget_data['output'].items():
#     cc += 1
#     print(type(out))
#     if not isinstance(out,str):
#         out = str(out).split(' ')[0]
#         print(id)
#         print(out)
#         print("注意！：")
#         print(type(out))
#     if cc == 500: break



        # print(forget_data_task1[x]['question'])
        # print(forget_data_task1[x]['answer'])

# print(len(forget_data['id']))
# idx = 1
# str_idx = str(idx)
# print(forget_data['id'][str_idx])
# print(retain_data['id']['0'])
#
# data_path11 = "/disk/disk_20T/share/semeval25/validation"
# forget_data11 = datasets.load_dataset(path=data_path11,data_files="forget.jsonl")['train'][0]
# retain_data11 = datasets.load_dataset(path=data_path11,data_files="retain.jsonl")['train'][0]
# print(forget_data11['id']['0'])
# print(retain_data11['id']['0'])