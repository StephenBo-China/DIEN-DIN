import tensorflow as tf
from tensorflow.keras import layers
from layers import AUGRU,attention
from activations import Dice,dice
from loss import AuxLayer
import utils

class DIEN(tf.keras.Model):
    def __init__(self, embedding_count_dict, embedding_dim_dict, embedding_features_list, user_behavior_features, activation="PReLU"):
        super(DIEN, self).__init__(embedding_count_dict, embedding_dim_dict, embedding_features_list, activation)
        """DIEN初始化model函数
        
        该函数在调用DIEN时进行DIEN的Embedding层,GRU层,AUGRU层,全连接层的初始化操作

        Args:
            embedding_count_dict:string->int格式,该变量记录需要embedding各个特征的词典个数,即最大整数索引+ 1的大小;
            embedding_dim_dict:string->int格式,该变量记录需要embedding各个特征的输出维数,即密集嵌入的尺寸;
            embedding_features_list:list(string)格式,该变量记录DIEN中user_profile部分所有需要embedding的feature名称;
            user_behavior_features:list(string)格式,该变量记录DIEN中user_behavior与target_item部分所有需要embedding的feature名称
            activation:string格式,默认值"PReLU",该变量空值全连接层激活函数,”PReLU“->PReLU,"Dice"->Dice
        """
        #Init Embedding Layer
        self.embedding_dim_dict = embedding_dim_dict
        self.embedding_count_dict = embedding_count_dict
        self.embedding_layers = dict()
        for feature in embedding_features_list:
            self.embedding_layers[feature] = layers.Embedding(embedding_count_dict[feature], embedding_dim_dict[feature])
        #Init GRU Layer
        self.user_behavior_gru = layers.GRU(self.get_GRU_input_dim(embedding_dim_dict, user_behavior_features), return_sequences=True)
        #Init Attention Layer
        self.attention_layer = layers.Softmax()
        #Init Auxiliary Layer
        self.AuxNet = AuxLayer()
        #Init AUGRU Layer
        self.user_behavior_augru = AUGRU(self.get_GRU_input_dim(embedding_dim_dict, user_behavior_features))
        #Init Fully Connection Layer
        self.fc = tf.keras.Sequential()
        self.fc.add(layers.BatchNormalization())
        self.fc.add(layers.Dense(200, activation="relu")) 
        if activation == "Dice":
            self.fc.add(Dice())
        elif activation == "dice":
            self.fc.add(dice(200))
        elif activation == "PReLU":
            self.fc.add(layers.PReLU(alpha_initializer='zeros', weights=None))
        self.fc.add(layers.Dense(80, activation="relu"))
        if activation == "Dice":
            self.fc.add(Dice()) 
        elif activation == "dice":
            self.fc.add(dice(80))
        elif activation == "PReLU":
            self.fc.add(layers.PReLU(alpha_initializer='zeros', weights=None))
        self.fc.add(layers.Dense(2, activation=None))

    def get_GRU_input_dim(self, embedding_dim_dict, user_behavior_features):
        rst = 0
        for feature in user_behavior_features:
            rst += embedding_dim_dict[feature]
        return rst

    def get_emb(self, user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list):
        user_profile_feature_embedding = dict()
        for feature in user_profile_list:
            data = user_profile_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            user_profile_feature_embedding[feature] = embedding_layer(data)
        
        target_item_feature_embedding = dict()
        for feature in user_behavior_list:
            data = target_item_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            target_item_feature_embedding[feature] = embedding_layer(data)
        
        click_behavior_embedding = dict()
        for feature in user_behavior_list:
            data = click_behavior_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            click_behavior_embedding[feature] = embedding_layer(data)
        
        # noclick_behavior_embedding = dict()
        # for feature in user_behavior_list:
        #     data = noclick_behavior_dict[feature]
        #     embedding_layer = self.embedding_layers[feature]
        #     noclick_behavior_embedding[feature] = embedding_layer(data)
        
        return utils.concat_features(user_profile_feature_embedding), utils.concat_features(target_item_feature_embedding), utils.concat_features(click_behavior_embedding)#, utils.concat_features(noclick_behavior_embedding)

    def auxiliary_loss(self, hidden_states, embedding_out):
        """Auxiliary Loss Function
        
        论文中包含的源代码aux loss是通过hidden state与点击序列concate和hidden state
        与展现序列concat后进一个全连接神经网络，通过softmax得到最终二分类结果与点击序列和展现序列求解log_loss的到最终aux loss。
        该方法只使用用户的点击序列。

        Args:
            hidden_states: gru产出的所有hidden state,从h(0)到h(n-1)
            embedding_out: gru输入的embedding特征,从e(1)到e(n)
        """
        click_input_ = tf.concat([hidden_states, embedding_out], -1)
        click_prop_ = self.AuxNet(click_input_)[:, :, 0]
        click_loss_ = - tf.reshape(tf.math.log(click_prop_), [-1, tf.shape(embedding_out)[1]])
        return tf.reduce_mean(click_loss_)
    
    def call(self, user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list):
        """输入batch训练数据, 调用DIEN初始化后的model进行一次前向传播

        调用该函数进行一次前向传播得到output, logit, aux_loss后,在自定义的训练函数内得出target_loss与final_loss后使用tensorflow中的梯度计算函数通过链式法则得到各层梯度后使用自定义优化器进行一次权重更新

        Args:
            user_profile_dict:dict:string->Tensor格式,记录user_profile部分的所有输入特征的训练数据;
            user_profile_list:list(string)格式,记录user_profile部分的所有特征名称;
            click_behavior_dict:dict:string->Tensor格式,记录user_behavior部分所有点击输入特征的训练数据;
            noclick_behavior_dict:dict:string->Tensor格式,记录user_behavior部分所有未点击输入特征的训练数据;
            target_item_dict:dict:string->Tensor格式,记录target_item部分输入特征的训练数据;
            user_behavior_list:list(string)Tensor格式,记录user_behavior部分的所有特征名称。
        """
        #Embedding Layer
        user_profile_embedding, target_item_embedding, click_behavior_emebedding = self.get_emb(user_profile_dict, user_profile_list, click_behavior_dict, target_item_dict, noclick_behavior_dict, user_behavior_list)
        #GRU Layer
        click_gru_emb = self.user_behavior_gru(click_behavior_emebedding)
        #noclick_gru_emb = self.user_behavior_gru(noclick_behavior_emebedding)
        #Auxiliary Loss
        aux_loss = self.auxiliary_loss(click_gru_emb[:, :-1, :], click_behavior_emebedding[:, 1:, :])
        #Attention Layer
        hist_attn = self.attention_layer(tf.matmul(tf.expand_dims(target_item_embedding, 1), click_gru_emb, transpose_b=True))
        #AUGRU Layer
        augru_hidden_state = tf.zeros_like(click_gru_emb[:, 0, :])
        for in_emb, in_att in zip(tf.transpose(click_gru_emb, [1, 0, 2]), tf.transpose(hist_attn, [2, 0, 1])):
            augru_hidden_state = self.user_behavior_augru(in_emb, augru_hidden_state, in_att)
        join_emb = tf.concat([augru_hidden_state, user_profile_embedding], -1)
        logit = tf.squeeze(self.fc(join_emb))
        output = tf.keras.activations.softmax(logit)
        return output, logit, aux_loss

class DIN(tf.keras.Model):
    def __init__(self, embedding_count_dict, embedding_dim_dict, embedding_features_list, user_behavior_features, activation="PReLU"):
        super(DIN, self).__init__(embedding_count_dict, embedding_dim_dict, embedding_features_list, user_behavior_features, activation)
        #Init Embedding Layer
        self.embedding_dim_dict = embedding_dim_dict
        self.embedding_count_dict = embedding_count_dict
        self.embedding_layers = dict()
        for feature in embedding_features_list:
            self.embedding_layers[feature] = layers.Embedding(embedding_count_dict[feature], embedding_dim_dict[feature])
        #DIN Attention+Sum pooling
        self.hist_at = attention(utils.get_input_dim(embedding_dim_dict, user_behavior_features))
        #Init Fully Connection Layer
        self.fc = tf.keras.Sequential()
        self.fc.add(layers.BatchNormalization())
        self.fc.add(layers.Dense(200, activation="relu")) 
        if activation == "Dice":
            self.fc.add(Dice())
        elif activation == "dice":
            self.fc.add(dice(200))
        elif activation == "PReLU":
            self.fc.add(layers.PReLU(alpha_initializer='zeros', weights=None))
        self.fc.add(layers.Dense(80, activation="relu"))
        if activation == "Dice":
            self.fc.add(Dice()) 
        elif activation == "dice":
            self.fc.add(dice(80))
        elif activation == "PReLU":
            self.fc.add(layers.PReLU(alpha_initializer='zeros', weights=None))
        self.fc.add(layers.Dense(2, activation=None))

    def get_emb_din(self, user_profile_dict, user_profile_list, hist_behavior_dict, target_item_dict, user_behavior_list):
        user_profile_feature_embedding = dict()
        for feature in user_profile_list:
            data = user_profile_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            user_profile_feature_embedding[feature] = embedding_layer(data)
        
        target_item_feature_embedding = dict()
        for feature in user_behavior_list:
            data = target_item_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            target_item_feature_embedding[feature] = embedding_layer(data)
        
        hist_behavior_embedding = dict()
        for feature in user_behavior_list:
            data = hist_behavior_dict[feature]
            embedding_layer = self.embedding_layers[feature]
            hist_behavior_embedding[feature] = embedding_layer(data)
        
        return utils.concat_features(user_profile_feature_embedding), utils.concat_features(target_item_feature_embedding), utils.concat_features(hist_behavior_embedding)
    
    def call(self, user_profile_dict, user_profile_list, hist_behavior_dict, target_item_dict, user_behavior_list, length):
        #Embedding Layer
        user_profile_embedding, target_item_embedding, hist_behavior_emebedding = self.get_emb_din(user_profile_dict, user_profile_list, hist_behavior_dict, target_item_dict, user_behavior_list)
        hist_attn_emb = self.hist_at(target_item_embedding, hist_behavior_emebedding, length)
        join_emb = tf.concat([user_profile_embedding, target_item_embedding, hist_attn_emb], -1)
        logit = tf.squeeze(self.fc(join_emb))
        output = tf.keras.activations.softmax(logit)
        return output, logit

if __name__ == "__main__":
    model = DIN(dict(), dict(), list(), list())
