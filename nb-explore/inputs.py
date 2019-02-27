import features
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import pandas as pd
import os
import helpers

def load_transient_labels(indir='../data/'):
    filename = 'transient_labels.csv'
    filepath = indir + filename
    df_cat = pd.read_csv(filepath)
    # Rename columns to match light curves
    df_cat = df_cat.rename(
        columns={'TransientID': 'ID', 'Classification': 'class'})
    df_cat.ID = 'TranID' + df_cat.ID.apply(str)
    df_cat = df_cat.set_index('ID')
    return df_cat


def top_transient_classes():
    return ['SN', 'CV', 'AGN', 'HPM', 'Blazar', 'Flare']


def assert_unique_copy_num(df_feats):
    unique_ids = df_feats.index.get_level_values('copy_num').unique()
    assert(len(unique_ids) == 1)
    assert(unique_ids[0] == 0)


def _assert_matrices_elements_are_contained_(in_from, in_to):
    for i in range(in_from.shape[0]):
        in_from_row = in_from[i]
        assert any((in_to[:] == in_from_row).all(1))


def _assert_matrices_elements_are_not_contained_(in_from, in_to):
    for i in range(in_from.shape[0]):
        in_from_row = in_from[i]
        assert not any((in_to[:] == in_from_row).all(1))


def assert_correct_input_generation(inputs, oversampled):
    assert(inputs is not None)
    assert(len(inputs) == 5)

    start_index = 0 if not oversampled else 1

    # Assert Correct Shapes
    for input_i in inputs[start_index:]:
        assert(input_i is not None)
        assert(len(input_i) == 3)
        for input_i_by_obs in input_i:
            assert(input_i_by_obs is not None)
            assert(len(input_i_by_obs) == 4)

    # Assert inputs Are the same for all feat_num in tasks
    i = 0
    for task_input_i in inputs[start_index:]:
        low_inputs, mid_inputs, hig_inputs = task_input_i

        # Compare X_train labels
        assert np.array_equal(low_inputs[0], mid_inputs[0][:, :20])
        assert np.array_equal(mid_inputs[0], hig_inputs[0][:, :26])

        # Compare X_test labels
        assert np.array_equal(low_inputs[2], mid_inputs[2][:, :20])
        assert np.array_equal(mid_inputs[2], hig_inputs[2][:, :26])

        # Compare y_train labels
        assert np.array_equal(low_inputs[1], mid_inputs[1])
        assert np.array_equal(mid_inputs[1], hig_inputs[1])
        
        print(i, low_inputs[1])
        i+=1

        # Compare y_test labels
        assert np.array_equal(low_inputs[3], mid_inputs[3])
        assert np.array_equal(mid_inputs[3], hig_inputs[3])

    # Assert Train Elements are not in Test Elements
    for task_input_i in inputs[start_index:]:
        for input_i_by_obs in [task_input_i[0], task_input_i[2]]:
            task_X_train, _, task_X_test, _ = input_i_by_obs
            _assert_matrices_elements_are_not_contained_(
                task_X_train, task_X_test)

    # Pre: Obtain Train and Test High-Features for Each Task
    sixt_hig_X_train, _, sixt_hig_X_test, _ = inputs[1][2]
    sevt_hig_X_train, _, sevt_hig_X_test, _ = inputs[2][2]
    sevc_hig_X_train, _, sevc_hig_X_test, _ = inputs[3][2]
    eigc_hig_X_train, _, eigc_hig_X_test, _ = inputs[4][2]

    # Assert all 6-Transients Elements are present in the Seven Transients Elements
    _assert_matrices_elements_are_contained_(
        sixt_hig_X_train, sevt_hig_X_train)
    _assert_matrices_elements_are_contained_(sixt_hig_X_test, sevt_hig_X_test)
    # Asser all 7-Transients Elements are present in the 8-Classes Elements
    _assert_matrices_elements_are_contained_(
        sevt_hig_X_train, eigc_hig_X_train)
    _assert_matrices_elements_are_contained_(sevt_hig_X_test, eigc_hig_X_test)
    # Assert all 7-Classes Elements are present in the 8-Classes Elements
    _assert_matrices_elements_are_contained_(
        sevc_hig_X_train, eigc_hig_X_train)
    _assert_matrices_elements_are_contained_(sevc_hig_X_test, eigc_hig_X_test)


def _assert_input_generation_replicability_(df_feats_t, df_feats_nt, num_obs):
    in1 = _inputs_(df_feats_t, df_feats_nt, num_obs, verbose=False)
    in2 = _inputs_(df_feats_t, df_feats_nt, num_obs, verbose=False)
    for in_index in [0, 1]:
        for task_i, _ in enumerate(in1[in_index]):
            if in_index == 1 and task_i == 0:
                continue
            for num_feats_i, _ in enumerate(in1[in_index][task_i]):
                for data_subset_i, _ in enumerate(in1[in_index][task_i][num_feats_i]):
                    assert np.array_equal(in1[in_index][task_i][num_feats_i][data_subset_i],
                                          in2[in_index][task_i][num_feats_i][data_subset_i])


def _input_dir_(task_index, oversampled, num_features, base_path='../data/inputs'):
    oversampled_name = 'oversampled' if oversampled else 'simple'
    task_names = ['binary', 'six_transients',
                  'seven_transients', 'seven_classes', 'eight_classes']
    output_dir = os.path.join(base_path,task_names[task_index], oversampled_name, num_features)

    return output_dir


def separate_by_number_of_features(df_feats):
    return df_feats[features.LOW], df_feats[features.MID], df_feats[features.HIGH]


def _filter_light_curves_(df, min_obs):
    df = df[df.ObsCount >= min_obs]
    assert df.ObsCount.min() >= min_obs
    return df


def _sort_by_indexes_(df):
    return df.sort_index(level=['copy_num', 'ID'])


def _remove_copies_(df):
    df = df.iloc[df.index.get_level_values('copy_num') == 0]
    unique_copynums = df.index.get_level_values('copy_num').unique()
    assert len(unique_copynums) == 1
    assert unique_copynums[0] == 0
    return df


def _shuffle_by_ids_(df):
    np.random.seed(42)
    df = df.reindex(np.random.permutation(df.index))
    return df


def _non_transient_ids_subsample_(df_nt, size):
    np.random.seed(43)
    nont_ids_all = df_nt.index.get_level_values('ID').unique()
    nont_ids_subset = np.random.choice(nont_ids_all, size=size, replace=False)
    df_nt = df_nt.loc[nont_ids_subset]
    assert num_unique_transient_ids(df_nt) == size
    return df_nt


def _split_into_feats_and_labels_(df):
    # Obtain Labels
    y = df['Class'].values
    # Drop Labels
    X = df.drop('Class', axis=1).values
    return X, y


def _generate_inputs_from_features_tuples_(dfs_train, dfs_test):
    assert len(dfs_train) == len(dfs_test)

    inputs_by_num_feat = tuple()
    for i, _ in enumerate(dfs_train):
        X_train, y_train = _split_into_feats_and_labels_(dfs_train[i])
        X_test, y_test = _split_into_feats_and_labels_(dfs_test[i])
        current_in = (X_train, y_train, X_test, y_test)
        inputs_by_num_feat += (current_in,)
    assert len(inputs_by_num_feat) == 3
    return inputs_by_num_feat


def _to_numpy_inputs_(df_train, df_test):
    # # Shuffle
    df_train = _shuffle_by_ids_(df_train)
    df_test = _shuffle_by_ids_(df_test)
    # Separate into inputs by numbers of features
    dfs_train = separate_by_number_of_features(df_train)
    dfs_test = separate_by_number_of_features(df_test)
    assert len(dfs_train) == 3
    assert len(dfs_test) == 3

    return _generate_inputs_from_features_tuples_(dfs_train, dfs_test)


def _largest_class_size_(df_feats_t):
    max = df_feats_t.loc[
        df_feats_t.index.get_level_values('copy_num') == 0
    ].groupby('Class')['ObsCount'].count().max()
    assert max > 0
    return max


def num_unique_transient_ids(df_feats_t):
    return len(df_feats_t.index.get_level_values('ID').unique())


def _split_train_test_(df, test_ids):
    df_train = df.loc[~df.index.isin(test_ids, level='ID')].copy()
    df_test = df.loc[df.index.isin(test_ids, level='ID')].copy()
    assert ~df_train.index.isin(test_ids, level='ID').all()
    assert df_test.index.isin(test_ids, level='ID').all()
    return df_train, df_test


def _test_ids_(df):

    test_ids = []
    for clss in df.Class.unique():
        df_class = df[df.Class == clss]
        # print(clss, num_unique_transient_ids(df_class))
        sampled_ids = df_class.sample(
            frac=0.3, random_state=42).index.get_level_values('ID').unique()
        # print(len(sampled_ids))
        test_ids.extend(sampled_ids)
    assert type(test_ids) is list
    return test_ids


def _rename_class_(df, name):
    df = df.copy()
    df['Class'] = name

    assert len(df.Class.unique()) == 1 and df.Class.unique()[0] == name
    return df


def _separate_transients_(df_t):
    main_classes = top_transient_classes()
    df_t = df_t.copy()

    df_other = df_t[~df_t.Class.isin(main_classes)]
    df_t = df_t[df_t.Class.isin(main_classes)]

    assert sorted(df_t.Class.unique()) != sorted(df_other.Class.unique())
    return df_t, df_other


def _multiclass_dataframes_(df_t, df_nt):
    # Generate Subsets
    df_multi_tr, df_multi_ot = _separate_transients_(df_t)
    num_max_transients = _largest_class_size_(df_t)
    df_multi_nt = _non_transient_ids_subsample_(df_nt, num_max_transients)
    # Rename Classes
    df_multi_ot = _rename_class_(df_multi_ot, 'Other')
    df_multi_nt = _rename_class_(df_multi_nt, 'Non-Transient')
    return df_multi_tr, df_multi_ot, df_multi_nt


def _balance_main_transients_(df_t, multi_t_ids, largest_class_size):
    main_classes = df_t.Class.unique()
    assert len(main_classes) == len(top_transient_classes()) and sorted(
        main_classes) == sorted(top_transient_classes())

    # Filter out Test Samples
    df_test = df_t[df_t.index.isin(multi_t_ids, level='ID')]
    df_test = _remove_copies_(df_test)
    df_train_all = df_t[~df_t.index.isin(multi_t_ids, level='ID')]
    # Create empty DataFrame
    df_train_balanced = pd.DataFrame()
    for t_class in main_classes:
        # Find "largest_class_size" elements of current class
        df_class = df_train_all[df_train_all.Class ==
                                t_class][: largest_class_size]
        assert(df_class.shape[0] == largest_class_size)
        # Append to empty df
        df_train_balanced = df_train_balanced.append(df_class)

    assert(df_train_balanced.shape[0] ==
           largest_class_size * len(main_classes))

    return df_train_balanced.append(df_test)


def _balance_other_transients_(df_other, multi_ot_ids, largest_class_size):
    # Filter out Test Samples
    df_test = df_other[df_other.index.isin(multi_ot_ids, level='ID')]
    df_test = _remove_copies_(df_test)
    df_train_all = df_other[~df_other.index.isin(multi_ot_ids, level='ID')]
    # Subsample Others
    df_train_balanced = df_train_all[:largest_class_size]

    return df_train_balanced.append(df_test)


def _balance_non_transients_(df_nt, multi_nt_ids, largest_class_size):

    # Obtain same as for non_oversampled
    df_train_nonbalanced = _non_transient_ids_subsample_(
        df_nt, largest_class_size)
    # Filter out Test Samples
    df_train = df_train_nonbalanced[~df_train_nonbalanced.index.isin(
        multi_nt_ids, level='ID')]
    filtered_size = num_unique_transient_ids(df_train)
    filtered_ids = df_train.index.get_level_values('ID').unique()
    assert filtered_size == largest_class_size - len(multi_nt_ids)
    # Sample new missing lightcurves to complete largest_class_size
    # by first obtaining DataFrame without elements in test or in filtered
    df_temp = df_nt[~df_nt.index.isin(multi_nt_ids, level='ID')]
    df_temp = df_temp[~df_temp.index.isin(filtered_ids, level='ID')]

    missing_size = largest_class_size - filtered_size
    assert missing_size > 0

    df_train_missing = _non_transient_ids_subsample_(df_temp, missing_size)
    assert num_unique_transient_ids(df_train_missing) == missing_size

    # Obtain Test Elements DataFrame
    df_test = df_nt[df_nt.index.isin(multi_nt_ids, level='ID')]

    # Merge Train, missing_train and test
    df_final = df_train.append(df_train_missing).append(df_test)
    assert df_final.shape[0] == df_final.drop_duplicates().shape[0]
    return df_final


def _multiclass_dataframes_balanced_(df_t, df_nt, multi_t_ids, multi_ot_ids, multi_nt_ids):
    # Separate Transients into Main, Other
    df_t, df_other = _separate_transients_(df_t)
    # Balance Transients DataFrames
    largest_class_size = _largest_class_size_(df_t)
    df_t = _balance_main_transients_(df_t, multi_t_ids, largest_class_size)
    df_other = _balance_other_transients_(
        df_other, multi_ot_ids, largest_class_size)
    df_nt = _balance_non_transients_(df_nt, multi_nt_ids, largest_class_size)
    # Rename Classes
    df_other = _rename_class_(df_other, 'Other')
    df_nt = _rename_class_(df_nt, 'Non-Transient')
    return df_t, df_other, df_nt


def _binaryclass_dataframes_(df_t, df_nt):
    df_t = _rename_class_(df_t, 'Transient')
    df_nt = _rename_class_(df_nt, 'Non-Transient')
    return df_t, df_nt


def _inputs_(df_feats_t_all, df_feats_nt_all, num_obs, verbose=True):
    assert num_obs in [5, 10]
    assert df_feats_t_all is not df_feats_nt_all
    if verbose:
        print('Generating Inputs for {} Observations'.format(num_obs))

    # ------- PREPARATION --------

    # Filter LCs with less than 5 observations
    df_feats_t_all = _filter_light_curves_(df_feats_t_all, num_obs)
    df_feats_nt_all = _filter_light_curves_(df_feats_nt_all, num_obs)
    # Subsample Non-Transient DataFrame to same size as transients
    num_transients = num_unique_transient_ids(df_feats_t_all)
    df_feats_nt_all = _non_transient_ids_subsample_(
        df_feats_nt_all, num_transients)

    # ------ NON OVERSAMPLED -------

    # 1. Prepare Datasets
    df_feats_t = _remove_copies_(df_feats_t_all)
    df_feats_t = _sort_by_indexes_(df_feats_t)
    df_feats_nt = _remove_copies_(df_feats_nt_all)
    df_feats_nt = _sort_by_indexes_(df_feats_nt)

    # 2. Obtain Binary and MultiClass DataFrames
    df_binary_t, df_binary_nt = _binaryclass_dataframes_(
        df_feats_t, df_feats_nt)

    df_multi_t, df_multi_ot, df_multi_nt = _multiclass_dataframes_(
        df_feats_t, df_feats_nt)

    # 3. Obtain test ids for each task
    binary_t_ids = _test_ids_(df_binary_t)
    binary_nt_ids = _test_ids_(df_binary_nt)
    multi_t_ids = _test_ids_(df_multi_t)
    multi_ot_ids = _test_ids_(df_multi_ot)
    multi_nt_ids = _test_ids_(df_multi_nt)

    # 4. Obtain Inputs for each task
    binary_in = _binary_inputs_(
        df_binary_t, df_binary_nt, binary_t_ids, binary_nt_ids)
    six_transients_in = _six_transients_inputs_(df_multi_t, multi_t_ids)
    seven_transients_in = _seven_transients_inputs_(
        df_multi_t, df_multi_ot, multi_t_ids, multi_ot_ids)
    seven_class_in = _seven_classes_inputs_(
        df_multi_t, df_multi_nt, multi_t_ids, multi_nt_ids)
    eight_class_in = _eight_classes_inputs_(
        df_multi_t, df_multi_ot, df_multi_nt, multi_t_ids, multi_ot_ids, multi_nt_ids)

    regular_inputs = [binary_in, six_transients_in,
                      seven_transients_in, seven_class_in, eight_class_in]

    # ------ OVERSAMPLED ---------

    # 1. Prepare Datasets
    df_feats_t = _sort_by_indexes_(df_feats_t_all)
    df_feats_nt = _sort_by_indexes_(df_feats_nt_all)

    # 2. Obtain MultiClass DataFrames
    df_multi_t, df_multi_ot, df_multi_nt = _multiclass_dataframes_balanced_(
        df_feats_t, df_feats_nt, multi_t_ids, multi_ot_ids, multi_nt_ids)

    six_transients_in = _six_transients_inputs_(df_multi_t, multi_t_ids)
    seven_transients_in = _seven_transients_inputs_(
        df_multi_t, df_multi_ot, multi_t_ids, multi_ot_ids)
    seven_class_in = _seven_classes_inputs_(
        df_multi_t, df_multi_nt, multi_t_ids, multi_nt_ids)
    eight_class_in = _eight_classes_inputs_(
        df_multi_t, df_multi_ot, df_multi_nt, multi_t_ids, multi_ot_ids, multi_nt_ids)

    balanced_inputs = [None, six_transients_in,
                       seven_transients_in, seven_class_in, eight_class_in]

    return regular_inputs, balanced_inputs


def _binary_inputs_(df_t, df_nt, binary_t_ids, binary_nt_ids):
    # 1. Split
    dfs_t_split = _split_train_test_(df_t, binary_t_ids)
    dfs_nt_split = _split_train_test_(df_nt, binary_nt_ids)
    # 2. Merge Test and Train DFs
    df_train = dfs_t_split[0].append(dfs_nt_split[0])
    df_test = dfs_t_split[1].append(dfs_nt_split[1])
    # 3. Convert to numpy
    return _to_numpy_inputs_(df_train, df_test)


def _six_transients_inputs_(df_t, t_ids):
    # 1. Split
    df_train, df_test = _split_train_test_(df_t, t_ids)
    # 2. Convert to numpy
    return _to_numpy_inputs_(df_train, df_test)


def _seven_transients_inputs_(df_t, df_o, t_ids, o_ids):
    # 1. Split
    dfs_t_split = _split_train_test_(df_t, t_ids)
    dfs_other_split = _split_train_test_(df_o, o_ids)
    # 2. Merge Test and Train DFs
    df_train = dfs_t_split[0].append(dfs_other_split[0])
    df_test = dfs_t_split[1].append(dfs_other_split[1])
    # 3. Convert to numpy
    return _to_numpy_inputs_(df_train, df_test)


def _seven_classes_inputs_(df_t, df_nt, t_ids, nt_ids):
    # 1. Split
    dfs_t_split = _split_train_test_(df_t, t_ids)
    dfs_nt_split = _split_train_test_(df_nt, nt_ids)
    # 2. Merge Test and Train DFs
    df_train = dfs_t_split[0].append(dfs_nt_split[0])
    df_test = dfs_t_split[1].append(dfs_nt_split[1])
    # 4. Convert to numpy
    return _to_numpy_inputs_(df_train, df_test)


def _eight_classes_inputs_(df_t, df_o, df_nt, t_ids, o_ids, nt_ids):
    # 1. Split
    dfs_t_split = _split_train_test_(df_t, t_ids)
    dfs_ot_split = _split_train_test_(df_o, o_ids)
    dfs_nt_split = _split_train_test_(df_nt, nt_ids)
    # 2. Merge Test and Train DFs
    df_train = dfs_t_split[0].append(dfs_ot_split[0]).append(dfs_nt_split[0])
    df_test = dfs_t_split[1].append(dfs_ot_split[1]).append(dfs_nt_split[1])
    # 3. Convert to numpy
    return _to_numpy_inputs_(df_train, df_test)


def generate_and_save_all_inputs(df_feats_t_all, df_feats_nt_all):
    num_obs_list = [5, 10]
    for num_obs in num_obs_list:
        _assert_input_generation_replicability_(
            df_feats_t_all, df_feats_nt_all, num_obs)
        # Generate Inputs For Current Parameter Combination
        regular_in, oversampled_in = _inputs_(
            df_feats_t_all, df_feats_nt_all, num_obs)
        assert_correct_input_generation(regular_in, oversampled=False)
        assert_correct_input_generation(oversampled_in, oversampled=True)
        save_tasks_inputs(regular_in, num_obs, oversampled=False)
        save_tasks_inputs(oversampled_in, num_obs, oversampled=True)


def save_tasks_inputs(tasks_inputs, num_obs, oversampled):

    num_features_list = [len(features.LOW) - len(features.BASE),
                         len(features.MID) - len(features.BASE),
                         len(features.HIGH) - len(features.BASE)]

    for i, task_i_in in enumerate(tasks_inputs):
        if task_i_in is None:
            continue
        for j, num_obs_in in enumerate(task_i_in):
            num_features = num_features_list[j]
            output_dir = _input_dir_(i, oversampled, num_features)
            helpers.make_dir_if_not_exists(output_dir)
            output_path = output_dir + '{}'.format(num_obs)

            np.save(output_path, num_obs_in)
            print('Successfully saved: ' + output_path)

# LOADING


def load_binary(num_obs, num_features, oversampled, scaler=None):
    return _load_task_inputs_(0, num_obs, num_features, oversampled, scaler)


def load_six_transients(num_obs, num_features, oversampled, scaler=None):
    return _load_task_inputs_(1, num_obs, num_features, oversampled, scaler)


def load_seven_transients(num_obs, num_features, oversampled, scaler=None):
    return _load_task_inputs_(2, num_obs, num_features, oversampled, scaler)


def load_seven_classes(num_obs, num_features, oversampled, scaler=None):
    return _load_task_inputs_(3, num_obs, num_features, oversampled, scaler)


def load_eight_classes(num_obs, num_features, oversampled, scaler=None):
    return _load_task_inputs_(4, num_obs, num_features, oversampled, scaler)


def _load_task_inputs_(task_index, num_obs, num_features, oversampled, scaler):
    input_dir = _input_dir_(task_index, oversampled, num_features)
    input_path = input_dir + '{}.npy'.format(num_obs)
    X_train, y_train, X_test, y_test = np.load(input_path)

    # Binarise Target Labels
    if task_index == 0:
        # lb =
        # print(y_train[:5])
        y_train = np.squeeze(label_binarize(
            y_train, ['Non-Transient', 'Transient']))
        # print(y_train[:5])
        y_test = np.squeeze(label_binarize(
            y_test, ['Non-Transient', 'Transient']))

    if scaler is not None:
        fit_scaler = scaler().fit(X_train)
        X_train = fit_scaler.transform(X_train)
        X_test = fit_scaler.transform(X_test)

    return X_train, y_train, X_test, y_test
