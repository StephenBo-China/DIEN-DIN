import pandas as pd
import tensorflow as tf
import numpy as np

def get_embedding_features_list():
    embedding_features_list = ["cate", "brand", "cms_segid", "cms_group", 
                           "gender", "age", "pvalue", "shopping", 
                           "occupation", "user_class_level"]
    return embedding_features_list
    
def get_user_behavior_features():
    user_behavior_features = ["cate", "brand"]
    return user_behavior_features

def get_embedding_count(feature, embedding_count):
    return embedding_count[feature].values[0]

def get_embedding_count_dict(embedding_features_list, embedding_count):
    embedding_count_dict = dict()
    for feature in embedding_features_list:
        embedding_count_dict[feature] = get_embedding_count(feature, embedding_count)
    embedding_count_dict["brand"] = 500000
    embedding_count_dict["cate"] = 501578
    embedding_count_dict["gender"] = 3
    embedding_count_dict["pvalue"] = 10
    embedding_count_dict["shopping"] = 4
    embedding_count_dict["occupation"] = 5
    embedding_count_dict["user_class_level"] = 5
    return embedding_count_dict

def get_embedding_dim_dict(embedding_features_list):
    embedding_dim_dict = dict()
    for feature in embedding_features_list:
        embedding_dim_dict[feature] = 64
    return embedding_dim_dict

def get_data():
    train_data = pd.read_csv("./data/train.csv", sep = "\t")
    train_data = train_data.fillna(0)
    train_data = train_data[train_data["guide_dien_final_train_data.click_cate"] != 0]
    train_data = train_data[train_data["guide_dien_final_train_data.click_brand"] != 0]
    test_data = pd.read_csv("./data/test.csv", sep = "\t")
    test_data = test_data.fillna(0)
    test_data = test_data[test_data["guide_dien_final_train_data.click_cate"] != 0]
    test_data = test_data[test_data["guide_dien_final_train_data.click_brand"] != 0]
    embedding_count = pd.read_csv("./data/embedding_count.csv")
    return train_data, test_data, embedding_count

def get_normal_data(data, col):
    return data[col].values

def get_sequence_data(data, col):
    rst = []
    max_length = 0
    for i in data[col].values:
        temp = len(list(map(eval,i[1:-1].split(","))))
        if temp > max_length:
            max_length = temp

    for i in data[col].values:
        temp = list(map(eval,i[1:-1].split(",")))
        padding = np.zeros(max_length - len(temp))
        rst.append(list(np.append(np.array(temp), padding)))
    return rst

def get_length(data, col):
    rst = []
    for i in data[col].values:
        temp = len(list(map(eval,i[1:-1].split(","))))
        rst.append(temp)
    return rst

def convert_tensor(data):
    return tf.convert_to_tensor(data)

def get_batch_data(data, min_batch, batch=100):
    # batch_data = None
    # if min_batch + batch <= len(data):
    #     batch_data = data.loc[min_batch:min_batch + batch - 1]
    # else:
    #     batch_data = data.loc[min_batch:]
    batch_data = data.sample(n=batch)
    click = get_normal_data(batch_data, "guide_dien_final_train_data.clk")
    #no_click = get_normal_data(batch_data, "guide_dien_final_train_data.nonclk")
    #label = [click, no_click]
    #label = click
    target_cate = get_normal_data(batch_data, "guide_dien_final_train_data.cate_id")
    target_brand = get_normal_data(batch_data, "guide_dien_final_train_data.brand")
    cms_segid = get_normal_data(batch_data, "guide_dien_final_train_data.cms_segid")
    cms_group = get_normal_data(batch_data, "guide_dien_final_train_data.cms_group_id")
    gender = get_normal_data(batch_data, "guide_dien_final_train_data.final_gender_code")
    age = get_normal_data(batch_data, "guide_dien_final_train_data.age_level")
    pvalue = get_normal_data(batch_data, "guide_dien_final_train_data.pvalue_level")
    shopping = get_normal_data(batch_data, "guide_dien_final_train_data.shopping_level")
    occupation = get_normal_data(batch_data, "guide_dien_final_train_data.occupation")
    user_class_level = get_normal_data(batch_data, "guide_dien_final_train_data.new_user_class_level")
    hist_brand_behavior_clk = get_sequence_data(batch_data, "guide_dien_final_train_data.click_brand")
    hist_cate_behavior_clk = get_sequence_data(batch_data, "guide_dien_final_train_data.click_cate")
    hist_brand_behavior_show = get_sequence_data(batch_data, "guide_dien_final_train_data.show_brand")
    hist_cate_behavior_show = get_sequence_data(batch_data, "guide_dien_final_train_data.show_cate")
    #reshape_len = convert_tensor(label).numpy().shape[1]
    clk_length = get_length(batch_data, "guide_dien_final_train_data.click_brand")
    show_length = get_length(batch_data, "guide_dien_final_train_data.show_brand")
    return tf.one_hot(click, 2), convert_tensor(target_cate), convert_tensor(target_brand), convert_tensor(cms_segid), convert_tensor(cms_group), convert_tensor(gender), convert_tensor(age), convert_tensor(pvalue), convert_tensor(shopping), convert_tensor(occupation), convert_tensor(user_class_level), convert_tensor(hist_brand_behavior_clk), convert_tensor(hist_cate_behavior_clk), convert_tensor(hist_brand_behavior_show), convert_tensor(hist_cate_behavior_show), min_batch + batch, clk_length, show_length

def get_test_data(data):
    batch_data = data.head(150)
    #batch_data = data.sample(n = 50)
    click = get_normal_data(batch_data, "guide_dien_final_train_data.clk")
    target_cate = get_normal_data(batch_data, "guide_dien_final_train_data.cate_id")
    target_brand = get_normal_data(batch_data, "guide_dien_final_train_data.brand")
    cms_segid = get_normal_data(batch_data, "guide_dien_final_train_data.cms_segid")
    cms_group = get_normal_data(batch_data, "guide_dien_final_train_data.cms_group_id")
    gender = get_normal_data(batch_data, "guide_dien_final_train_data.final_gender_code")
    age = get_normal_data(batch_data, "guide_dien_final_train_data.age_level")
    pvalue = get_normal_data(batch_data, "guide_dien_final_train_data.pvalue_level")
    shopping = get_normal_data(batch_data, "guide_dien_final_train_data.shopping_level")
    occupation = get_normal_data(batch_data, "guide_dien_final_train_data.occupation")
    user_class_level = get_normal_data(batch_data, "guide_dien_final_train_data.new_user_class_level")
    hist_brand_behavior_clk = get_sequence_data(batch_data, "guide_dien_final_train_data.click_brand")
    hist_cate_behavior_clk = get_sequence_data(batch_data, "guide_dien_final_train_data.click_cate")
    hist_brand_behavior_show = get_sequence_data(batch_data, "guide_dien_final_train_data.show_brand")
    hist_cate_behavior_show = get_sequence_data(batch_data, "guide_dien_final_train_data.show_cate")
    clk_length = get_length(batch_data, "guide_dien_final_train_data.click_brand")
    show_length = get_length(batch_data, "guide_dien_final_train_data.show_brand")
    return tf.one_hot(click, 2), convert_tensor(target_cate), convert_tensor(target_brand), convert_tensor(cms_segid), convert_tensor(cms_group), convert_tensor(gender), convert_tensor(age), convert_tensor(pvalue), convert_tensor(shopping), convert_tensor(occupation), convert_tensor(user_class_level), convert_tensor(hist_brand_behavior_clk), convert_tensor(hist_cate_behavior_clk), convert_tensor(hist_brand_behavior_show), convert_tensor(hist_cate_behavior_show), clk_length, show_length