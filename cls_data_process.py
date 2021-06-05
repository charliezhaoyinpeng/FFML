import pandas as pd
import numpy as np
import copy, math, os, random
from scipy import ndimage


def process_communities_and_crime():
    data_path = r'data/raw/communities_and_crime.csv'
    df = pd.read_csv(data_path)
    task_names = np.unique(df['state'].values)
    # print(task_names, len(task_names))
    tasks_0, tasks_1 = [], []
    for task_name in task_names:
        task = df[df['state'] == task_name]
        del task['state']
        ys = task['y'].values
        ans = copy.deepcopy(ys)
        median = np.median(ys)
        for i in range(len(ys)):
            if ys[i] > median:
                ans[i] = 1
            else:
                ans[i] = 0
        task['y'] = ans
        # del task['x23']
        task = rotate_feature(task, 2500, 100)
        task_norm = (task - task.mean()) / task.std()
        task_norm['z'] = task['z']
        task_norm['y'] = task['y']
        task_norm_0 = task_norm[task_norm['y'] == 0]
        task_norm_1 = task_norm[task_norm['y'] == 1]
        tasks_0.append(task_norm_0)
        tasks_1.append(task_norm_1)
    save_folder = r'data/communities_and_crime'
    for i in range(1, len(task_names) + 1):
        save = save_folder + '/task' + str(i)
        if not os.path.exists(save):
            os.makedirs(save)
        if tasks_0[i - 1].isnull().values.any():
            print("task_neg %s contains missing values." % (i))
        tasks_0[i - 1].to_csv(save + '/task' + str(i) + '_neg.csv')
        if tasks_1[i - 1].isnull().values.any():
            print("task_pos %s contains missing values." % (i))
        tasks_1[i - 1].to_csv(save + '/task' + str(i) + '_pos.csv')


def process_adult():
    data_path = r'data/raw/adult.csv'
    df = pd.read_csv(data_path)
    df['x2'] = df['x2'].astype('category')
    df['x4'] = df['x4'].astype('category')
    df['x6'] = df['x6'].astype('category')
    df['x7'] = df['x7'].astype('category')
    df['x8'] = df['x8'].astype('category')
    df['x9'] = df['x9'].astype('category')
    # print(df.dtypes)
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    task_names = np.unique(df['location'].values)
    # print(df.dtypes)
    # print(task_names, len(task_names))
    tasks_0, tasks_1 = [], []
    for task_name in task_names:
        task = df[df['location'] == task_name]
        del task['location']
        task = rotate_feature(task, 2500, 16)
        task_norm = (task - task.mean()) / task.std()
        task_norm['z'] = task['z']
        task_norm['y'] = task['y']
        task_norm_0 = task_norm[task_norm['y'] == 0]
        task_norm_1 = task_norm[task_norm['y'] == 1]
        tasks_0.append(task_norm_0)
        tasks_1.append(task_norm_1)
    save_folder = r'data/adult'
    for i in range(1, len(task_names) + 1):
        save = save_folder + '/task' + str(i)
        if not os.path.exists(save):
            os.makedirs(save)
        if tasks_0[i - 1].isnull().values.any():
            print("task_neg %s contains missing values." % (i))
        tasks_0[i - 1].to_csv(save + '/task' + str(i) + '_neg.csv')
        if tasks_1[i - 1].isnull().values.any():
            print("task_pos %s contains missing values." % (i))
        tasks_1[i - 1].to_csv(save + '/task' + str(i) + '_pos.csv')


def process_bank():
    data_path = r'data/raw/bankmarketing.csv'
    df = pd.read_csv(data_path)
    df['x2'] = df['x2'].astype('category')
    df['x3'] = df['x3'].astype('category')
    df['x4'] = df['x4'].astype('category')
    df['x5'] = df['x5'].astype('category')
    df['x6'] = df['x6'].astype('category')
    df['x7'] = df['x7'].astype('category')
    df['x12'] = df['x12'].astype('category')
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    task_names = np.unique(df['date'].values)
    # print(task_names, len(task_names))
    tasks_0, tasks_1 = [], []
    for task_name in task_names:
        task = df[df['date'] == task_name]
        del task['date']
        task = rotate_feature(task, 2500, 16)
        task_norm = (task - task.mean()) / task.std()
        task_norm['z'] = task['z']
        task_norm['y'] = task['y']
        task_norm_0 = task_norm[task_norm['y'] == 0]
        task_norm_1 = task_norm[task_norm['y'] == 1]
        tasks_0.append(task_norm_0)
        tasks_1.append(task_norm_1)
    save_folder = r'data/bank'
    for i in range(1, len(task_names) + 1):
        save = save_folder + '/task' + str(i)
        if not os.path.exists(save):
            os.makedirs(save)
        if tasks_0[i - 1].isnull().values.any():
            print("task_neg %s contains missing values." % (i))
        tasks_0[i - 1].to_csv(save + '/task' + str(i) + '_neg.csv')
        if tasks_1[i - 1].isnull().values.any():
            print("task_pos %s contains missing values." % (i))
        tasks_1[i - 1].to_csv(save + '/task' + str(i) + '_pos.csv')


def rotate(inputs, angle, d_feature):
    sqrt = int(np.sqrt(d_feature))
    return ndimage.rotate(inputs.reshape(sqrt, sqrt), angle, reshape=False).reshape(1, d_feature)


def rotate_feature(task, nums, d_feature):
    task = task.reset_index(drop=True)
    indices = list(task.index)
    angles = list(range(1, 360))
    ans = copy.deepcopy(task)
    while len(ans) < nums:
        for index in indices:
            sample = task.loc[index].values
            X = sample[-d_feature:]
            y = sample[1]
            z = sample[0]
            rotated_X = rotate(X, random.choice(angles), d_feature)
            rotated_yX = np.insert(rotated_X, 0, y)
            rotated_zyX = np.insert(rotated_yX, 0, z)
            ans.loc[len(ans)] = rotated_zyX
    return ans


def cal_discrimination(input_zy):
    a_values = []
    b_values = []
    for line in input_zy:
        if line[0] == 0:
            a_values.append(line[1])
        elif line[0] == 1:
            b_values.append(line[1])
    if len(a_values) == 0:
        discrimination = sum(b_values) * 1.0 / len(b_values)
    elif len(b_values) == 0:
        discrimination = sum(a_values) * 1.0 / len(a_values)
    else:
        discrimination = sum(a_values) * 1.0 / len(a_values) - sum(b_values) * 1.0 / len(b_values)
    return abs(discrimination)


# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return math.sqrt(distance)


# Locate the most similar neighbors
# example: get_neighbors(yX, X[0], 3)
def get_neighbors(yX, target_row, num_neighbors):
    distances = list()
    for yX_row in yX:
        X_row = yX_row[1:]
        y = yX_row[0]
        dist = euclidean_distance(target_row, X_row)
        distances.append((y, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def cal_consistency(yX, num_neighbors):
    ans = 0
    for yX_row in yX:
        temp = 0
        target_row = yX_row[1:]
        target_y = yX_row[0]
        y_neighbors = get_neighbors(yX, target_row, num_neighbors)
        for y_neighbor in y_neighbors:
            temp += abs(target_y - y_neighbor)
        ans += temp
    return (1 - (ans * 1.0) / (len(yX) * num_neighbors))


def cal_dbc(input_zy):
    length = len(input_zy)
    z_bar = np.mean(input_zy[:, 0])
    dbc = 0
    for zy in input_zy:
        dbc += (zy[0] - z_bar) * zy[1] * 1.0
    return abs(dbc / length)


def tasks_evaluation(save, num_neighbors):
    total_dbc = 0
    total_discrimination = 0
    total_consistency = 0
    tasks = [x[0] for x in os.walk(save)]
    for task_num in range(1, len(tasks)):
        df0 = pd.read_csv(save + '/task' + str(task_num) + '/task' + str(task_num) + '_neg.csv')
        df1 = pd.read_csv(save + '/task' + str(task_num) + '/task' + str(task_num) + '_pos.csv')
        df = pd.concat([df0, df1])
        zy = df[['z', 'y']].values
        yX = df[df.columns[2:]].values
        discrimination = cal_discrimination(zy)
        consistency = cal_consistency(yX, num_neighbors)
        dbc = cal_dbc(zy)
        print("task %s: dbc=%s, discrimination=%s, consistency=%s" % (task_num, dbc, discrimination, consistency))
        total_dbc += dbc
        total_discrimination += discrimination
        total_consistency += consistency
    print("#################################################################################################")
    print("Average dbc=%s" % (total_dbc / len(tasks)))
    print("Average discrimination=%s" % (total_discrimination / len(tasks)))
    print("Average consistency=%s" % (total_consistency / len(tasks)))


if __name__ == "__main__":
    # process communities and crime data set
    #########################################################################################################
    # tasks = process_communities_and_crime()
    # path = r'data/communities_and_crime'
    # tasks_evaluation(path, 3)
    #########################################################################################################

    # process adult data set
    #########################################################################################################
    # tasks = process_adult()
    # path = r'data/adult'
    # tasks_evaluation(path, 3)
    #########################################################################################################

    # process bank data set
    #########################################################################################################
    tasks = process_bank()
    path = r'data/bank'
    tasks_evaluation(path, 3)
    #########################################################################################################


