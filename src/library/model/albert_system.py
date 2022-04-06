"""ALBERT reservoir for short term analysis. 
Please specify the environmental variables "USE_BERT_ATTENTION_PROJECT" 
to "1" to obtain the attention values.
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import tensorflow as tf
import params_flow as pf
import collections

from pyutils.tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import backend as K

if os.environ['USE_BERT_ATTENTION_PROJECT'] == "0":
    import bert
    from bert import StockBertConfig
    from bert.layer import Layer
    from bert.embeddings import BertEmbeddingsLayer
    from bert.transformer import TransformerEncoderLayer
    USE_ATTENTION = False
else:
    from src.library import bert
    from src.library.bert import StockBertConfig
    from src.library.bert.layer import Layer
    from src.library.bert.embeddings import BertEmbeddingsLayer
    from src.library.bert.transformer import TransformerEncoderLayer
    USE_ATTENTION = True

from src.library.bert import StockBertConfig
from src.library.bert.layer import Layer
from src.library.bert.embeddings import BertEmbeddingsLayer
from src.library.bert.transformer import TransformerEncoderLayer

from src.library.model.albert_tokenization import Tokenizer
from src.library.utils import metrics

param_list = [
    ("embedding_size", 128),
    ("num_hidden_groups", 1),
    ("net_structure_type", 0),
    ("gap_size", 0),
    ("num_memory_blocks", 0),
    ("inner_group_num", 1),
    ("down_scale_factor", 1),
]

for _name, _param in param_list:
    setattr(StockBertConfig, _name, _param)


def get_visible_devices():
    return tf.config.experimental.list_physical_devices('GPU')


def set_visible_devices(selected_id=None):
    gpu_list = tf.config.experimental.list_physical_devices('GPU')
    try:
        if type(selected_id) == list:
            gpu_list = [gpu_list[_id] for _id in selected_id]
        for _gpu in gpu_list:
            print("Device: {}".format(_gpu))
        tf.config.experimental.set_visible_devices(gpu_list, 'GPU')
    except RuntimeError as e:
        print(e)


def update_rec_intermediate(rec, vals, callback):
    """
    Update values for recording.

    Parameters:
    ----------
    rec: dict.
        key: str. intermediate_type.
        value: np.nedarray or tf.Tensor. intermediate_value.
    vals: dict.
        key: str. intermediate_type.
        value: tf.Tensor. intermediate_value.
    """
    for key in vals.keys():
        additional_val = callback(vals[key])
        rec[key].append(additional_val)


class AlbertSystem(keras.models.Model):
    def __init__(self,
                 model_name="albert_base",
                 version="2",
                 seq_len=128,
                 position=0,
                 pretrained_weights=False,
                 *args,
                 **kwargs):
        '''
        Wrapper for ALBERT model implemented based on bert-for-tf2
        ALBERT: https://arxiv.org/abs/1909.11942
        bert-for-tf2: https://github.com/kpe/bert-for-tf2/tree/master/bert

        Parameters:
        ----------
        model_name: str
            fetched from tensorflow-hub
            {"albert_base", "albert_large", "albert_xlarge", "albert_xxlarge"}
        version: str
            version of albert
            {"1", "2"}
        seq_len: int
            length of input sequence
            should be positive and less than "max_position_embeddings"
        position: int
            position of print line(tqdm)
        pretrained_weights: bool
            whether to use pre-trained model's parameters.
        '''
        super().__init__(*args, **kwargs)
        fetch_dir = ".models/{}-v{}".format(model_name, version)
        self.albert_dir = bert.fetch_tfhub_albert_model(model_name,
                                                        fetch_dir,
                                                        version=version)
        self.params = bert.albert_params(model_name)
        self.albert = bert.BertModelLayer.from_params(self.params,
                                                      name='albert')
        self.num_attention_heads = self.params["num_heads"]
        self.seq_len = seq_len
        assert 0 < self.seq_len <= self.params["max_position_embeddings"]
        self.build([(None, self.seq_len), (None, self.seq_len)])
        # for weight in self.albert.weights:
        #     print(weight.name)
        if pretrained_weights:
            skipped_weight_value_tuples = \
                bert.load_albert_weights(self.albert, self.albert_dir)
            assert 0 == len(skipped_weight_value_tuples)
            print("Weights successfully have been loaded")

        self.position = position
        self.tokenizer = Tokenizer(self.albert_dir)
        self.data_type = np.float32

    def call(self, inputs, training=False):
        '''
        Default call function
        You can call ALBERT model by model(inputs)

        Parameters:
        ----------
        inputs: [ndarray(np.uint), ndarray(np.uint)] or ndarray(np.uint)
        '''
        return self.albert(inputs, training=False)

    def set_input(self,
                  input_ids,
                  token_type_ids,
                  attention_mask=None,
                  training=None):
        inputs = [input_ids, token_type_ids]
        if attention_mask is None:
            attention_mask = self.albert.embeddings_layer.compute_mask(inputs)
        self._embedding_state = self.albert.embeddings_layer(
            inputs, mask=attention_mask, training=training)
        self.mask = attention_mask
        self._init_state = self.x
        self._attention_state = None

    def reset(self, state=None):
        if state is not None:
            self._init_state = np.array(state)
        self.x = self._init_state

    @property
    def x(self):
        return self._embedding_state.numpy().astype(np.float64)

    @property
    def att(self):  # attention
        return self._attention_state.numpy().astype(np.float64)

    @x.setter
    def x(self, value):
        assert value.ndim == 3
        assert value.shape[-1] == self.params["hidden_size"]
        self._embedding_state = tf.convert_to_tensor(value, dtype=tf.float32)

    @att.setter
    def att(self, value):
        assert value.ndim == 4
        assert value.shape[-1] <= self.params["max_position_embeddings"]
        self._attention_state = tf.convert_to_tensor(value, dtype=tf.float32)

    def step(self,
             step_num=1,
             mask=None,
             training=None,
             save_state=False,
             callback=None,
             callback_gpu=None,
             get_intermediate=False,
             get_attention_heads=[],
             verbose=True,
             prefix=""):

        if get_intermediate:
            if not USE_ATTENTION:
                raise ValueError(
                    "When USE_BERT_ATTENTION_PROJECT is 0, you should be specify get_intermediate=False"
                )
            return self.step_with_intermediate(
                step_num,
                mask=None,
                training=training,
                save_state=save_state,
                callback=callback,
                callback_gpu=callback_gpu,
                get_attention_heads=get_attention_heads,
                verbose=verbose,
                prefix="")
        else:
            # simple iteration.
            if USE_ATTENTION:
                raise ValueError(
                    "When USE_BERT_ATTENTION_PROJECT is 1, you should be specify get_intermediate=True"
                )
            return self.simple_step(step_num,
                                    mask=None,
                                    training=training,
                                    save_state=save_state,
                                    callback=callback,
                                    callback_gpu=callback_gpu,
                                    verbose=verbose,
                                    prefix="")

    def simple_step(self,
                    step_num=1,
                    mask=None,
                    training=None,
                    save_state=False,
                    callback=None,
                    callback_gpu=None,
                    verbose=True,
                    prefix=""):
        iterator = range(step_num)
        if verbose:
            iterator = tqdm(iterator,
                            leave=False,
                            desc="{}step()".format(prefix))
        if save_state:
            assert callback is not None or callback_gpu is not None
            if callback is not None:
                rec_net = np.zeros((step_num, *callback(self.x).shape),
                                   dtype=self.data_type)
            elif callback_gpu is not None:
                rec_net = np.zeros(
                    (step_num, *callback_gpu(self._embedding_state).shape),
                    dtype=self.data_type)
        for _ in iterator:
            self._embedding_state = \
                self.albert.encoders_layer.shared_layer(
                    self._embedding_state,
                    mask=mask,
                    training=training)
            if save_state:
                if callback is not None:
                    rec_net[_] = callback(self.x)
                else:
                    step_result = callback_gpu(self._embedding_state)
                    rec_net[_] = step_result

        if save_state:
            return rec_net

    def step_with_intermediate(self,
                               step_num=1,
                               mask=None,
                               training=None,
                               save_state=False,
                               callback=None,
                               callback_gpu=None,
                               get_attention_heads=[],
                               verbose=True,
                               prefix=""):
        iterator = range(step_num)
        valid_callback = callback if callback is not None else callback_gpu
        if verbose:
            iterator = tqdm(iterator,
                            leave=False,
                            desc="{}step()".format(prefix))
        if not USE_ATTENTION:
            raise ValueError("Please set environmental value \
                                USE_BERT_ATTENTION_PRJECT='1' \
                                    If you want to use attention info.")

        if save_state:
            assert valid_callback is not None
            rec_net = np.zeros((step_num, *valid_callback(self.x).shape))
            rec_att = np.zeros(
                (step_num, valid_callback(self.x).shape[0],
                 len(get_attention_heads), self.seq_len, self.seq_len),
                dtype=self.data_type)
        rec_intermediate = collections.defaultdict(list)
        for _ in iterator:
            self._embedding_state, intermediate_vals = \
                self.albert.encoders_layer.shared_layer(
                    self._embedding_state,
                    mask=mask,
                    training=training,
                    get_intermediate=True)
            self._attention_state = intermediate_vals["attention_probs"]
            if save_state:
                rec_net[_] = valid_callback(self.x)
                rec_att[_] = valid_callback(
                    self._attention_state.numpy())[:, get_attention_heads]

                for key in intermediate_vals.keys():
                    additional_val = valid_callback(intermediate_vals[key])
                    rec_intermediate[key].append(additional_val)
        if save_state:
            return rec_net, rec_att, rec_intermediate

    def maximum_lyapunov(self,
                         tau,
                         T,
                         cutoff=False,
                         save_state=False,
                         save_one_dim=True,
                         save_one_token=False,
                         th=0.0,
                         epsilon=1e-6,
                         debug=False,
                         prefix1="",
                         prefix2="",
                         callback_gpu=None,
                         callback=None):
        '''
        Calculating maximum Lyapunov exponents for each batches.
        Please call "set_input" or "reset" to initialize the state.

        Parameters:
        ----------
        tau: int
        T: int
        cutoff: bool. Whether to stop calculation if whole lyapnov exponents is under
            threshold value "th".
        th: float. Threshold value.
        epsilon: float
        debug: bool. Whether to check the difference expanding.
        save_state: bool
            Whether or not to save the intermediate state.
        save_one_dim : bool, optional
            Whether or not to save only one dimension of intermediate states, by default True
        save_one_token : bool , optional
            Whether or not to save vector of only first word, by default False

        Returns:
        lyap_list: np.ndarray(np.float)
            list of maximum lyapunov exponents for each batch
            lyap_list.shape: (int(T/tau), batch_size)
        '''
        x_init = self.x  # (batch_size, seq_len, hidden_dim)
        batch_size = x_init.shape[0]
        pert = []
        for _ in range(batch_size):
            _pert = np.random.randn(*x_init.shape[1:])
            _pert *= epsilon / np.linalg.norm(_pert)
            pert.append(_pert)
        if save_state:
            assert callback is not None or callback_gpu is not None
            if callback is not None and save_one_dim:
                rec_net = np.zeros((T, batch_size))
            elif callback is not None and save_one_token:
                rec_net = np.zeros((T, batch_size, callback(self.x).shape[-1]))
            elif callback is not None:
                rec_net = np.zeros((T, *callback(self.x).shape))
            elif callback_gpu is not None and save_one_dim:
                rec_net = np.zeros((T, batch_size))
            elif callback_gpu is not None and save_one_token:
                rec_net = np.zeros(
                    (T, batch_size,
                     callback_gpu(self._embedding_state).shape[-1]))
            elif callback_gpu is not None:
                rec_net = np.zeros(
                    (T, *callback_gpu(self._embedding_state).shape))
        self.x = np.concatenate([x_init, x_init + pert], axis=0)
        total = int(T / tau)
        lyap_list = np.zeros((total, batch_size))
        description = "{}lyap()".format(prefix1)
        if debug:
            diff_record = np.zeros([T, batch_size])
        else:
            diff_record = None
        for _ in tqdm(range(total), desc=description, leave=False):
            _rec_net = self.step(tau,
                                 verbose=True,
                                 callback=callback,
                                 save_state=save_state or debug,
                                 callback_gpu=callback_gpu,
                                 prefix=prefix2,
                                 training=False)
            if debug:
                diff_record[_ * tau:(_ + 1) * tau] = np.linalg.norm(
                    _rec_net[:, batch_size:] - _rec_net[:, :batch_size],
                    axis=(-2, -1))
            _state = self.x
            if save_state and not (save_one_dim or save_one_token):
                rec_net[_ * tau:(_ + 1) * tau] = _rec_net[:, :batch_size]
            elif save_state and save_one_dim:
                rec_net[_ * tau:(_ + 1) * tau] = _rec_net[:, :batch_size, 0, 0]
            elif save_state and save_one_token:
                rec_net[_ * tau:(_ + 1) * tau] = _rec_net[:, :batch_size, 0, :]
            _diff = _state[batch_size:] - _state[:batch_size]
            _norm = tf.norm(_diff, axis=(1, 2))
            lyap_list[_] = tf.math.log(_norm / epsilon) / tau
            _state[batch_size:] = _state[:batch_size]
            for _ in range(batch_size):
                _state[batch_size + _] += (epsilon / _norm[_]) * _diff[_]
            self.x = _state
            if cutoff and np.where(lyap_list[_] < th, 0, 1).sum() == 0:
                break
        self.reset()
        if save_state:
            return lyap_list, diff_record, rec_net
        else:
            return lyap_list, diff_record, None

    def _calc_std(self, callback_gpu=None, callback=None):
        """Calculate standard deviation between tokens.
        """
        if callback_gpu is not None:
            _std = tf.math.reduce_std(self._embedding_state,
                                      axis=1,
                                      keepdims=False,
                                      name=None)
            return callback_gpu(tf.math.reduce_mean(_std, axis=-1))
        else:
            return callback(self.x.std(axis=1).mean(axis=-1))

    def calc_sync(self,
                  step_num=1,
                  training=None,
                  save_state=False,
                  callback=None,
                  callback_gpu=None,
                  verbose=True,
                  prefix=""):
        """calculate standard deviation between tokens.

        Parameters:
        =========
        step_num: int.

        Returns:
        =========
        std_record: ndarray.shape=(num_step, batch_size).
            The records of standard deviation.
        rec_net: ndarray. shape=(num_step, batch_size, max_seq_len, hidden_size).
            The records of network. If save_state=False, it returns None.
        """
        iterator = range(step_num)
        if verbose:
            iterator = tqdm(iterator,
                            leave=False,
                            desc="{}step()".format(prefix))
        assert callback is not None or callback_gpu is not None
        if callback is not None:
            std_record = np.zeros(
                (step_num + 1, callback(self._embedding_state).shape[0]))
        else:
            std_record = np.zeros(
                (step_num + 1, callback_gpu(self._embedding_state).shape[0]))
        std_record[0] = self._calc_std(callback=callback,
                                       callback_gpu=callback_gpu)
        if save_state:
            if callback is not None:
                rec_net = np.zeros((step_num, *callback(self.x).shape))
            elif callback_gpu is not None:
                rec_net = np.zeros(
                    (step_num, *callback_gpu(self._embedding_state).shape))
        for _ in iterator:
            self._embedding_state = \
                self.albert.encoders_layer.shared_layer(
                    self._embedding_state,
                    mask=self.mask,
                    training=training)
            std_record[_ + 1] = self._calc_std(callback=callback,
                                               callback_gpu=callback_gpu)
            if save_state:
                if callback is not None:
                    rec_net[_] = callback(self.x)
                else:
                    step_result = callback_gpu(self._embedding_state)
                    rec_net[_] = step_result

        if save_state:
            return std_record, rec_net
        else:
            return std_record, None

    def input_sentence(self, sentences, max_seq_len=16):
        """Make and set batch input for AlbertSystem

        Parameters:
        =========
        sentences: list of str.
        max_seq_len: int. default 16.
        """
        batch_inputs = collections.defaultdict(list)
        for sentence in sentences:
            inputs, _ = self.tokenizer.tokenize_text([sentence],
                                                     max_seq_len=max_seq_len,
                                                     return_pieces=False)
            for k, v in inputs.items():
                inputs[k] = np.array(v, dtype=np.int64).reshape(1, -1)
                batch_inputs[k].append(inputs[k].copy())
        for k, v in batch_inputs.items():
            batch_inputs[k] = np.concatenate(batch_inputs[k], axis=0)
        self.set_input(**batch_inputs)

    def calc_effective_dimension(self,
                                 sentences,
                                 max_seq_len=16,
                                 save_state=True,
                                 step_num=500,
                                 save_dir="./out/effective_dimension/",
                                 gpu_idxes=[0, 1]):
        effective_dimensions = np.zeros(step_num, dtype=np.float32)
        if gpu_idxes[0] == -1:
            self.input_sentence(sentences, max_seq_len=max_seq_len)
        else:
            with tf.device(f'/device:GPU:{gpu_idxes[0]}'):
                self.input_sentence(sentences, max_seq_len=max_seq_len)
        for step in tqdm(range(step_num)):
            if gpu_idxes[0] == -1:
                rec_net = self.step(step_num=1,
                                    callback_gpu=lambda x: x,
                                    save_state=True)
            else:
                with tf.device(f'/device:GPU:{gpu_idxes[0]}'):
                    rec_net = self.step(step_num=1,
                                        callback_gpu=lambda x: x,
                                        save_state=True)
            X = rec_net.reshape(len(sentences), -1)
            if save_state:
                save_path = os.path.join(save_dir, f"step{step:04d}.npy")
                np.save(save_path, X)
            effective_dimensions[step] = metrics.effective_dimension(
                X, gpu_idx=gpu_idxes[1])
        return effective_dimensions
