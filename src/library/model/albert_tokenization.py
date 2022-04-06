#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import six
import numpy as np
import unicodedata
import sentencepiece as spm

SPIECE_UNDERLINE = u"▁".encode("utf-8")


def preprocess_text(inputs, remove_space=True, lower=False):
    """preprocess data by removing extra space and normalize data."""
    outputs = inputs
    if remove_space:
        outputs = " ".join(inputs.strip().split())

    if six.PY2 and isinstance(outputs, str):
        try:
            outputs = six.ensure_text(outputs, "utf-8")
        except UnicodeDecodeError:
            outputs = six.ensure_text(outputs, "latin-1")

    outputs = unicodedata.normalize("NFKD", outputs)
    outputs = "".join([c for c in outputs if not unicodedata.combining(c)])
    if lower:
        outputs = outputs.lower()

    return outputs


def encode_pieces(sp_model, text, return_unicode=True, sample=False):
    """turn sentences into word pieces."""

    if not sample:
        pieces = sp_model.EncodeAsPieces(text)
    else:
        pieces = sp_model.SampleEncodeAsPieces(text, 64, 0.1)
    new_pieces = []
    for piece in pieces:
        if isinstance(piece, str):
            pass
        elif isinstance(text, bytes):
            piece = six.ensure_text(text, "utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
        if len(piece) > 1 and piece[-1] == "," and piece[-2].isdigit():
            cur_pieces = sp_model.EncodeAsPieces(
                six.ensure_binary(piece[:-1]).replace(SPIECE_UNDERLINE, b""))
            if piece[0] != SPIECE_UNDERLINE and \
                    cur_pieces[0][0] == SPIECE_UNDERLINE:
                if len(cur_pieces[0]) == 1:
                    cur_pieces = cur_pieces[1:]
                else:
                    cur_pieces[0] = cur_pieces[0][1:]
            cur_pieces.append(piece[-1])
            new_pieces.extend(cur_pieces)
        else:
            new_pieces.append(piece)

    return new_pieces


def encode_ids(sp_model, text, sample=False, return_pieces=False):
    pieces = encode_pieces(sp_model, text, return_unicode=False, sample=sample)
    ids = [sp_model.PieceToId(piece) for piece in pieces]
    if return_pieces:
        return ids, pieces
    else:
        return ids


def load_tokenizer(model_dir):
    spm_model = os.path.join(model_dir, "assets", "30k-clean.model")
    sp = spm.SentencePieceProcessor()
    do_lower_case = True
    sp.load(spm_model)
    return sp


def tokenize_text(text, sp_model, lower=True, return_pieces=False):
    processed_text = preprocess_text(text, lower=lower)
    return encode_ids(sp_model, processed_text, return_pieces=return_pieces)


class Tokenizer(object):
    def __init__(self, model_dir):
        self.sp = load_tokenizer(model_dir)
        self.cls_id = self.sp.piece_to_id("[CLS]")
        self.sep_id = self.sp.piece_to_id("[SEP]")
        self.pad_id = self.sp.piece_to_id("<pad>")

    def tokenize_text(
            self, texts, specified_ids=None, lower=True,
            return_pieces=False, use_cls=True, use_sep=True,
            max_seq_len=128):
        """
        Args :
            texts : list of text.
            specified_ids : list of id.
            lower : bool.
            return_pieces : bool. Check the tokens.
        """
        assert len(texts) <= 2, "must be less than 3"
        cls_id = self.sp.piece_to_id("[CLS]")
        sep_id = self.sp.piece_to_id("[SEP]")
        ids = [cls_id]
        token_type_ids = [0]*max_seq_len
        attention_mask = [0]*max_seq_len

        if use_cls:
            pieces = ["[CLS]"]
        else:
            pieces = []
        if type(specified_ids) is list:
            ids += specified_ids
            pieces += [self.sp.IdToPiece(_id) for _id in specified_ids]
            attention_mask[:len(ids)] = 1
            token_type_ids[:ids.index(sep_id)] = 0
            token_type_ids[ids.index(sep_id):] = 1
        else:
            pre_pos = 0
            for idx, text in enumerate(texts):
                if return_pieces:
                    _ids, _pieces = tokenize_text(
                        text, self.sp, lower=lower,
                        return_pieces=True)
                    if use_sep:
                        pieces += _pieces + ["[SEP]"]
                    else:
                        pieces += _pieces
                else:
                    _ids = tokenize_text(
                        text, self.sp, lower=lower,
                        return_pieces=return_pieces)
                if max_seq_len - (1*use_sep) < len(ids)+len(_ids):
                    over_length = len(ids) + len(_ids) - max_seq_len + 1
                    _ids = _ids[:-over_length]
                if use_sep:
                    ids += _ids + [sep_id]
                else:
                    ids += _ids
                token_type_ids[pre_pos:len(ids)] = [idx]*(len(ids) - pre_pos)
                pre_pos = len(ids)
        attention_mask[:len(ids)] = [1]*len(ids)
        if len(ids) < max_seq_len:
            ids += [self.pad_id]*(max_seq_len - len(ids))
        inputs = {
            "input_ids": ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask
        }
        pieces = pieces[:max_seq_len-1] + ["[SEP]"]
        return inputs, pieces
