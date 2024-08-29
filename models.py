# -*-coding:utf-8-*-
import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertModel
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_bert import BertEmbeddings, BertEncoder, BertPooler

from modules import MAG, WA, WSA
from MI_estimators import MINE, InfoNCE


class BERTModel(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, model="WSA-BERT"):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.model = model

        if self.model == "WA-BERT":
            self.WA = WA(config.hidden_size, multimodal_config.beta_shift, multimodal_config.dropout_prob)
        elif self.model == "MAG-BERT":
            self.MAG = MAG(config.hidden_size, multimodal_config.beta_shift, multimodal_config.dropout_prob)
        else:
            self.WSA = WSA(config.hidden_size, multimodal_config.beta_shift, multimodal_config.dropout_prob)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(
            attention_mask, input_shape, device
        )

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.size()
            encoder_hidden_shape = (
                encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None


        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        # Early fusion with MAG/WA/WSA, that is, r=0
        if self.model == "WA-BERT":
            fused_embedding = self.WA(embedding_output, visual, acoustic)
        elif self.model == "MAG-BERT":
            fused_embedding = self.MAG(embedding_output, visual, acoustic)
        else:
            fused_embedding = self.WSA(embedding_output, visual, acoustic)

        encoder_outputs = self.encoder(
            fused_embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return pooled_output


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, model="WSA-BERT"):
        super().__init__(config)

        self.bert = BERTModel(config, multimodal_config, model)

        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(multimodal_config.dropout2)

        self.encoder = nn.Linear(config.hidden_size, multimodal_config.latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(multimodal_config.latent_dim, multimodal_config.hidden_size_pred),
            nn.ReLU(),
            nn.Dropout(multimodal_config.dropout3),
            nn.Linear(multimodal_config.hidden_size_pred, config.num_labels)
        )

        self.init_weights()

    def forward(
        self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):

        pooled_output = self.bert(
            input_ids,
            visual,
            acoustic,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = self.dropout1(pooled_output)
        encoder_output = self.dropout2(self.encoder(pooled_output))
        outputs = self.classifier(encoder_output)

        return outputs


class BERTEncoder(nn.Module):
    def __init__(self, hidden_size=768, latent_dim=128, dropout=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.encoder = nn.Linear(hidden_size, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        pooled_output = outputs[1]

        encoder_output = self.encoder(self.dropout(pooled_output))

        return encoder_output


class TextLSTMEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim):
        super().__init__()
        self.hid_dim = hid_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTMCell(emb_dim, hid_dim)
        self.fc1 = nn.Linear(hid_dim, hid_dim)

    def forward(self, x):
        x = x.transpose(0, 1)
        embeded = self.embedding(x)
        t = embeded.shape[0]
        n = embeded.shape[1]
        self.hx = torch.zeros(n, self.hid_dim).cuda()
        self.cx = torch.zeros(n, self.hid_dim).cuda()
        all_hs = []
        all_cs = []
        for i in range(t):
            self.hx, self.cx = self.lstm(embeded[i], (self.hx, self.cx))
            all_hs.append(self.hx)
            all_cs.append(self.cx)
        # last hidden layer last_hs is n*h
        last_hs = all_hs[-1]
        last_hs = self.fc1(last_hs)
        return last_hs


class encoderLSTM(nn.Module):
    def __init__(self, d, h): #, n_layers, bidirectional, dropout):
        super(encoderLSTM, self).__init__()
        self.lstm = nn.LSTMCell(d, h)
        self.fc1 = nn.Linear(h, h)
        self.h = h

    def forward(self, x):
        # x is n*t*h
        x = x.transpose(0, 1) # transpose x to shape t*n*h
        t = x.shape[0]
        n = x.shape[1]
        self.hx = torch.zeros(n, self.h).cuda()
        self.cx = torch.zeros(n, self.h).cuda()
        all_hs = []
        all_cs = []
        for i in range(t):
            self.hx, self.cx = self.lstm(x[i], (self.hx, self.cx))
            all_hs.append(self.hx)
            all_cs.append(self.cx)
        # last hidden layer last_hs is n*h
        last_hs = all_hs[-1]
        last_hs = self.fc1(last_hs)
        return last_hs


class decoderLSTM(nn.Module):
    def __init__(self, h, d):
        super(decoderLSTM, self).__init__()
        self.lstm = nn.LSTMCell(h, h)
        self.fc1 = nn.Linear(h, d)
        self.d = d
        self.h = h

    def forward(self, hT, t):  # only embedding vector
        # x is n*d
        n = hT.shape[0]
        h = hT.shape[1]
        self.hx = torch.zeros(n, self.h).cuda()
        self.cx = torch.zeros(n, self.h).cuda()
        final_hs = []
        all_hs = []
        all_cs = []
        for i in range(t):
            if i == 0:
                self.hx, self.cx = self.lstm(hT, (self.hx, self.cx))
            else:
                self.hx, self.cx = self.lstm(all_hs[-1], (self.hx, self.cx))
            all_hs.append(self.hx)
            all_cs.append(self.cx)
            final_hs.append(self.hx.view(1, n, h))
        final_hs = torch.cat(final_hs, dim=0)
        all_recons = self.fc1(final_hs) # all_recons shape is T*N*d
        all_recons = all_recons.transpose(0, 1) # transpose to N*T*d
        return all_recons


class MultiEncoder(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, model='WSA-BERT', hidden_size=768, latent_dim=64, dropout=0.5):
        super().__init__(config)
        self.model = BERTModel(config, multimodal_config, model)
        self.encoder = nn.Linear(hidden_size, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                input_ids,
                visual,
                acoustic,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
    ):
        model_output = self.model(input_ids,
                        visual,
                        acoustic,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        position_ids=position_ids,
                        head_mask=head_mask,
                        inputs_embeds=inputs_embeds,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
        )
        encoder_output = self.encoder(self.dropout(model_output))

        return encoder_output



class WSABERTRecon(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, model="WSA-BERT"):
        super().__init__(config)
        self.d_a, self.d_v = multimodal_config.orig_d_a, multimodal_config.orig_d_v
        zy_size = multimodal_config.zy_size
        zl_size = multimodal_config.zl_size
        za_size = multimodal_config.za_size
        zv_size = multimodal_config.zv_size
        zy_to_y_dropout = multimodal_config.zy_to_y_dropout

        output_dim_l = multimodal_config.output_dim_l
        hidden_size_l = multimodal_config.hidden_size_l
        dropout_l = multimodal_config.dropout_l

        hidden_size_multi = multimodal_config.hidden_size_multi
        dropout_multi = multimodal_config.dropout_multi

        label_dim = multimodal_config.label_dim

        if multimodal_config.language_encoder_use_lstm:
            self.encoder_l = TextLSTMEncoder(30522, 128, 64)
        else:
            self.encoder_l = BERTEncoder(hidden_size=hidden_size_l, latent_dim=zl_size, dropout=dropout_l)
        self.encoder_a = encoderLSTM(self.d_a, za_size)
        self.encoder_v = encoderLSTM(self.d_v, zv_size)

        self.decoder_l = decoderLSTM(zy_size + zl_size, output_dim_l)
        self.decoder_a = decoderLSTM(zy_size + za_size, self.d_a)
        self.decoder_v = decoderLSTM(zy_size + zv_size, self.d_v)

        self.mult_encoder = MultiEncoder(config, multimodal_config, model, hidden_size_multi, zy_size, dropout_multi)

        self.zy_to_y_fc1 = nn.Linear(zy_size+zl_size+za_size+zv_size, zy_size)
        self.zy_to_y_fc2 = nn.Linear(zy_size, label_dim)
        self.zy_to_y_dropout = nn.Dropout(zy_to_y_dropout)

    def forward(self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        # input_ids is n*t, x_a, x_v is n*t*d
        tl = input_ids.shape[1]
        ta = acoustic.shape[1]
        tv = visual.shape[1]

        zl = self.encoder_l(input_ids)
        za = self.encoder_a(acoustic)
        zv = self.encoder_v(visual)

        zy = self.mult_encoder(input_ids,
                               visual,
                               acoustic,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
        )

        zyzl = torch.cat([zy, zl], dim=1)
        zyza = torch.cat([zy, za], dim=1)
        zyzv = torch.cat([zy, zv], dim=1)

        zylav = torch.cat([zy, zl, za, zv], dim=1)

        x_l_hat = self.decoder_l(zyzl, tl)
        x_a_hat = self.decoder_a(zyza, ta)
        x_v_hat = self.decoder_v(zyzv, tv)

        y_hat = self.zy_to_y_fc2(self.zy_to_y_dropout(F.relu(self.zy_to_y_fc1(zylav))))
        return zl, za, zv, zy, x_l_hat, x_a_hat, x_v_hat, y_hat


class WSABERTReconMIM(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, model="WSA-BERT"):
        super().__init__(config)
        self.d_a, self.d_v = multimodal_config.orig_d_a, multimodal_config.orig_d_v
        zy_size = multimodal_config.zy_size
        zl_size = multimodal_config.zl_size
        za_size = multimodal_config.za_size
        zv_size = multimodal_config.zv_size
        zy_to_y_dropout = multimodal_config.zy_to_y_dropout

        output_dim_l = multimodal_config.output_dim_l
        hidden_size_l = multimodal_config.hidden_size_l
        dropout_l = multimodal_config.dropout_l

        hidden_size_multi = multimodal_config.hidden_size_multi
        dropout_multi = multimodal_config.dropout_multi

        label_dim = multimodal_config.label_dim

        if multimodal_config.language_encoder_use_lstm:
            self.encoder_l = TextLSTMEncoder(30522, 128, 64)
        else:
            self.encoder_l = BERTEncoder(hidden_size=hidden_size_l, latent_dim=zl_size, dropout=dropout_l)
        self.encoder_a = encoderLSTM(self.d_a, za_size)
        self.encoder_v = encoderLSTM(self.d_v, zv_size)

        self.decoder_l = decoderLSTM(zy_size + zl_size, output_dim_l)
        self.decoder_a = decoderLSTM(zy_size + za_size, self.d_a)
        self.decoder_v = decoderLSTM(zy_size + zv_size, self.d_v)

        self.mult_encoder = MultiEncoder(config, multimodal_config, model, hidden_size_multi, zy_size, dropout_multi)

        self.zy_to_y_fc1 = nn.Linear(zy_size+zl_size+za_size+zv_size, zy_size)
        self.zy_to_y_fc2 = nn.Linear(zy_size, label_dim)
        self.zy_to_y_dropout = nn.Dropout(zy_to_y_dropout)

    def forward(self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        # input_ids is n*t, x_a, x_v is n*t*d
        tl = input_ids.shape[1]
        ta = acoustic.shape[1]
        tv = visual.shape[1]

        zl = self.encoder_l(input_ids)
        za = self.encoder_a(acoustic)
        zv = self.encoder_v(visual)

        zy = self.mult_encoder(input_ids,
                               visual,
                               acoustic,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
        )

        zyzl = torch.cat([zy, zl], dim=1)
        zyza = torch.cat([zy, za], dim=1)
        zyzv = torch.cat([zy, zv], dim=1)

        zylav = torch.cat([zy, zl, za, zv], dim=1)

        x_l_hat = self.decoder_l(zyzl, tl)
        x_a_hat = self.decoder_a(zyza, ta)
        x_v_hat = self.decoder_v(zyzv, tv)

        y_hat = self.zy_to_y_fc2(self.zy_to_y_dropout(F.relu(self.zy_to_y_fc1(zylav))))

        return zl, za, zv, zy, x_l_hat, x_a_hat, x_v_hat, y_hat


class WSABERTReconMIMM(BertPreTrainedModel):
    def __init__(self, config, multimodal_config, model="WSA-BERT"):
        super().__init__(config)
        self.d_a, self.d_v = multimodal_config.orig_d_a, multimodal_config.orig_d_v
        zy_size = multimodal_config.zy_size
        zl_size = multimodal_config.zl_size
        za_size = multimodal_config.za_size
        zv_size = multimodal_config.zv_size
        zy_to_y_dropout = multimodal_config.zy_to_y_dropout

        output_dim_l = multimodal_config.output_dim_l
        hidden_size_l = multimodal_config.hidden_size_l
        dropout_l = multimodal_config.dropout_l

        hidden_size_mim = multimodal_config.hidden_size_mim

        hidden_size_multi = multimodal_config.hidden_size_multi
        dropout_multi = multimodal_config.dropout_multi

        label_dim = multimodal_config.label_dim

        if multimodal_config.language_encoder_use_lstm:
            self.encoder_l = TextLSTMEncoder(30522, 128, 64)
        else:
            self.encoder_l = BERTEncoder(hidden_size=hidden_size_l, latent_dim=zl_size, dropout=dropout_l)
        self.encoder_a = encoderLSTM(self.d_a, za_size)
        self.encoder_v = encoderLSTM(self.d_v, zv_size)

        self.decoder_l = decoderLSTM(zy_size + zl_size, output_dim_l)
        self.decoder_a = decoderLSTM(zy_size + za_size, self.d_a)
        self.decoder_v = decoderLSTM(zy_size + zv_size, self.d_v)

        self.mult_encoder = MultiEncoder(config, multimodal_config, model, hidden_size_multi, zy_size, dropout_multi)

        self.infonce_l = InfoNCE(self.d_l, zy_size, hidden_size_mim)
        self.infonce_a = InfoNCE(self.d_a, zy_size, hidden_size_mim)
        self.infonce_v = InfoNCE(self.d_v, zy_size, hidden_size_mim)

        self.zy_to_y_fc1 = nn.Linear(zy_size+zl_size+za_size+zv_size, zy_size)
        self.zy_to_y_fc2 = nn.Linear(zy_size, label_dim)
        self.zy_to_y_dropout = nn.Dropout(zy_to_y_dropout)

    def forward(self,
        input_ids,
        visual,
        acoustic,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        # input_ids is n*t, x_a, x_v is n*t*d
        tl = input_ids.shape[1]
        ta = acoustic.shape[1]
        tv = visual.shape[1]

        zl = self.encoder_l(input_ids)
        za = self.encoder_a(acoustic)
        zv = self.encoder_v(visual)

        zy = self.mult_encoder(input_ids,
                               visual,
                               acoustic,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask,
                               inputs_embeds=inputs_embeds,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
        )

        zyzl = torch.cat([zy, zl], dim=1)
        zyza = torch.cat([zy, za], dim=1)
        zyzv = torch.cat([zy, zv], dim=1)

        zylav = torch.cat([zy, zl, za, zv], dim=1)

        x_l_hat = self.decoder_l(zyzl, tl)
        x_a_hat = self.decoder_a(zyza, ta)
        x_v_hat = self.decoder_v(zyzv, tv)

        mi_l = self.infonce_l(input_ids, zy)
        mi_a = self.infonce_a(acoustic.mean(1), zy)
        mi_v = self.infonce_v(visual.mean(1), zy)

        y_hat = self.zy_to_y_fc2(self.zy_to_y_dropout(F.relu(self.zy_to_y_fc1(zylav))))

        return zl, za, zv, zy, x_l_hat, x_a_hat, x_v_hat, mi_l, mi_a, mi_v, y_hat