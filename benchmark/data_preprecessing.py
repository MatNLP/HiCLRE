path = './no_preprocessing/nyt10/nyt10_train.txt'
write_path = './nyt10/nyt10_train_with_sentIndex.txt'
f = open(path, "r")
write_swap = open(write_path,"w")
print('begin to read and write...')
line_list = f.readlines()
for iter in range(len(line_list)):
    line = line_list[iter].rstrip()
    if len(line) > 0:
        line_dict = eval(line)
        line_dict['sentence_index'] = iter
        write_swap.write(str(line_dict) + '\n')
print('done')
f.close()
write_swap.close()