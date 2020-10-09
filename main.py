import tensorflow as tf
from tensorflow.keras import layers
from layers import AUGRU
from activations import Dice
import pandas as pd
from model import DIEN
import alibaba_data_reader as data_reader

def train_one_step(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list, label, optimizer, model, alpha, loss_metric):
        with tf.GradientTape() as tape:
            output, logit, aux_loss = model(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list)
            target_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=tf.cast(label, dtype=tf.float32)))
            final_loss = target_loss + alpha * aux_loss
            print("[Train Step] aux_loss=" + str(aux_loss.numpy()) + ", target_loss=" + str(target_loss.numpy()) + ", final_loss=" + str(final_loss.numpy()))
        gradient = tape.gradient(final_loss, model.trainable_variables)
        clip_gradient, _ = tf.clip_by_global_norm(gradient, 5.0)
        optimizer.apply_gradients(zip(clip_gradient, model.trainable_variables))
        loss_metric(final_loss)

def get_train_data(label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show):
    user_profile_dict = {
        "cms_segid": cms_segid,
        "cms_group": cms_group,
        "gender": gender,
        "age": age,
        "pvalue": pvalue,
        "shopping": shopping,
        "occupation": occupation,
        "user_class_level": user_class_level
    }
    user_profile_list = ["cms_segid", "cms_group", "gender", "age", "pvalue", "shopping", "occupation", "user_class_level"]
    user_behavior_list = ["brand", "cate"]
    click_behavior_dict = {
        "brand": hist_brand_behavior_clk,
        "cate": hist_cate_behavior_clk
    }
    noclick_behavior_dict = {
        "brand": hist_brand_behavior_show,
        "cate": hist_cate_behavior_show
    }
    target_item_dict = {
        "brand": target_cate,
        "cate": target_brand
    }
    return user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict

def main():
    train_data, test_data, embedding_count = data_reader.get_data()
    embedding_features_list = data_reader.get_embedding_features_list()
    user_behavior_features = data_reader.get_user_behavior_features()
    embedding_count_dict = data_reader.get_embedding_count_dict(embedding_features_list, embedding_count)
    embedding_dim_dict = data_reader.get_embedding_dim_dict(embedding_features_list)
    model = DIEN(embedding_count_dict, embedding_dim_dict, embedding_features_list, user_behavior_features)
    min_batch = 0
    batch = 100
    label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show, min_batch, clk_length, show_length = data_reader.get_batch_data(train_data, min_batch, batch = batch)
    user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict = get_train_data(label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show) 
    log_path = "./train_log/"
    train_summary_writer = tf.summary.create_file_writer(log_path)
    optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    loss_metric = tf.keras.metrics.Sum()
    auc_metric = tf.keras.metrics.AUC()
    alpha = 1
    epochs = 1
    for epoch in range(epochs):
        min_batch = 0
        for i in range(int(len(train_data) / batch)):
            label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show, min_batch, clk_length, show_length = data_reader.get_batch_data(train_data, min_batch, batch = batch)
            user_profile_dict, user_profile_list, user_behavior_list, click_behavior_dict, noclick_behavior_dict, target_item_dict = get_train_data(label, target_cate, target_brand, cms_segid, cms_group, gender, age, pvalue, shopping, occupation, user_class_level, hist_brand_behavior_clk, hist_cate_behavior_clk, hist_brand_behavior_show, hist_cate_behavior_show)
            train_one_step(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list, label, optimizer, model, alpha, loss_metric)


if __name__ == "__main__":
    print(tf.__version__)
    print("GPU Available: ", tf.test.is_gpu_available())
    main()