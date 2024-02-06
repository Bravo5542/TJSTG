import torch

import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange, repeat


def batch_organize(out_match_posi, out_match_nega):
    # audio B 512
    # posi B 512
    # nega B 512

    out_match = torch.zeros(out_match_posi.shape[0] * 2, out_match_posi.shape[1])
    batch_labels = torch.zeros(out_match_posi.shape[0] * 2)
    for i in range(out_match_posi.shape[0]):
        out_match[i * 2, :] = out_match_posi[i, :]
        out_match[i * 2 + 1, :] = out_match_nega[i, :]
        batch_labels[i * 2] = 1
        batch_labels[i * 2 + 1] = 0
    
    return out_match, batch_labels

# Question
class QstEncoder(nn.Module):

    def __init__(self, qst_vocab_size, word_embed_size, embed_size, num_layers, hidden_size):  # (93, 512, 512, 1, 512)

        super(QstEncoder, self).__init__()
        self.word2vec = nn.Embedding(qst_vocab_size, word_embed_size)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers)
        self.fc = nn.Linear(2*num_layers*hidden_size, embed_size)     # 2 for hidden and cell states
       
       
        self.fc1 = nn.Linear(num_layers*hidden_size, embed_size)
        self.fc2 = nn.Linear(embed_size, embed_size) 
        self.attn = nn.MultiheadAttention(512, 4, dropout=0.2)
    


    def forward(self, question):

        qst_vec = self.word2vec(question)                             # [batch_size, max_qst_length=14, word_embed_size=512]
        qst_vec = self.tanh(qst_vec)
        qst_vec = qst_vec.transpose(0, 1)                             # [max_qst_length=14, batch_size, word_embed_size=512]
        self.lstm.flatten_parameters()
        qst_lstm, (hidden, cell) = self.lstm(qst_vec)                        # [num_layers=1, batch_size, hidden_size=512] [L,B,C]
        qst_feature = torch.cat((hidden, cell), 2)                    # [num_layers=1, batch_size, 2*hidden_size=1024]
        

        feat_att, att_weights = self.attn(cell, qst_lstm, qst_lstm, attn_mask=None, key_padding_mask=None) #[14,32,512]
        viw_id = torch.max(att_weights,dim=-1)[1] #[32,1,14]->[32,1]
        viwt = qst_lstm.index_select(0,viw_id.squeeze()) #[32,32,512]
        viw = torch.diagonal(viwt,dim1=0,dim2=1) #[512，32]  
        viw = viw.transpose(0,1)
        viwt = viw.unsqueeze(0) #[1,32,512]
        viw = self.fc2(F.relu(self.fc1(viwt))) 
        
        
        qst_feature = qst_feature.transpose(0, 1)                     # [batch_size, num_layers=1, 2*hidden_size=1024]
        qst_feature = qst_feature.reshape(qst_feature.size()[0], -1)  # [batch_size, 2*num_layers*hidden_size=1024]
        qst_feature = self.tanh(qst_feature)
        qst_feature = self.fc(qst_feature)                            # [batch_size, embed_size]

        return qst_feature, viw

class AVQA_Fusion_Net(nn.Module):

    def __init__(self):
        super(AVQA_Fusion_Net, self).__init__()

        # for features
        self.fc_a1 = nn.Linear(128, 512)
        self.fc_a2 = nn.Linear(512,512)

        self.fc_a1_pure =  nn.Linear(128, 512)
        self.fc_a2_pure=nn.Linear(512,512)
        
        self.fc_avq = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)
        
        
        self.a_gru = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True, dropout=0.1)
        self.v_gru = nn.LSTM(512, 256, 1, batch_first=True, bidirectional=True, dropout=0.1)
       


        self.linearav1 = nn.Linear(512, 512)
        self.dropoutav1 = nn.Dropout(0.2) 
        self.dropoutav2 = nn.Dropout(0.2) 
        self.linearav2 = nn.Linear(512, 512)
        self.normav = nn.LayerNorm(512)


        
        self.attn_av = nn.MultiheadAttention(512, 4, dropout=0.1) 


        # question
        self.question_encoder = QstEncoder(93, 512, 512, 1, 512)
        self.tanh = nn.Tanh()
        self.fc_ans = nn.Linear(512, 42)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_gl=nn.Linear(1024,512)

      
        # combine
        self.fc1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 2)
        self.relu4 = nn.ReLU()

        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        
        self.qvfc = nn.Linear(512, 256)
        
       

      


    def forward(self, audio, visual_posi, visual_nega, question):
        '''
            input question shape:    [B, T] T = 10
            input audio shape:       [B, T, C]
            input visual_posi shape: [B, T, C, H, W]
            input visual_nega shape: [B, T, C, H, W]
        '''

        ## question features
        qst_feature, target = self.question_encoder(question) #[B, Embed]->[B, 512]
        xq = qst_feature.unsqueeze(0) # [1, B, 512]
        

        ## audio features  [2*B*T, 128]
        audio_feat = F.relu(self.fc_a1(audio))
        audio_feat = self.fc_a2(audio_feat)  
        audio_feat_pure = audio_feat # [B, 10, 512]
        B, T, C = audio_feat.size()             # [B, T, C]
        audio_feat = audio_feat.view(B*T, C)    # [B*T, C]->[10B, 512]


        ## visual posi [2*B*T, C, H, W]
        Bs, T, C, H, W = visual_posi.size()                      # [B, T, C, H, W]->[B, 10, 512, 14, 14]
        temp_visual = visual_posi.view(Bs*T, C, H, W)            # [B*T, C, H, W]->[10B, 512, 14, 14]
        v_feat = self.avgpool(temp_visual)                      # [B*T, C, 1, 1]->[10B, 512, 1, 1]
        visual_feat_before_grounding_posi = v_feat.squeeze()    # [B*T, C]->[10B, 512]

        (B, C, H, W) = temp_visual.size()
        v_feat = temp_visual.view(B, C, H * W)                      # [B*T, C, HxW]->[10B, 512, 196]
        v_feat = v_feat.permute(0, 2, 1)                            # [B, HxW, C]->[10B, 196, 512]
        visual_feat_posi = nn.functional.normalize(v_feat, dim=2)   # [B, HxW, C]->[10B, 196, 512]Normalize channels, Batchnorm?

        ## text-visual attention
        q_text = target # [1, B, 512]
        q_text = q_text.repeat(10,1,1)
        q_text = q_text.view(Bs*T, C)
        q_text_tt = q_text.unsqueeze(-1)
        q_text_tt = nn.functional.normalize(q_text_tt, dim=1)
        x1_it = torch.matmul(visual_feat_posi,q_text_tt).squeeze()

        x1_p = F.softmax(x1_it, dim=-1).unsqueeze(-2) 

        
        ## audio-visual grounding posi
        audio_feat_aa = audio_feat.unsqueeze(-1)                        # [B*T, C, 1]->[10B, 512, 1]
        audio_feat_aa = nn.functional.normalize(audio_feat_aa, dim=1)   # [B*T, C, 1]->[10B, 512, 1]Normalize channels
        x2_va = torch.matmul(visual_feat_posi, audio_feat_aa).squeeze() # [B*T, HxW]->[10B, 196]audio-visual similarity matrix 
  
        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [B*T, 1, HxW]->[10B, 1, 196]audio-visual similarity-probability matrix
        
        x_p = x1_p+x2_p
        tvmask = x_p>0.01 # ori:0.01
        x_p = x_p*tvmask 
        x_p = F.softmax(x_p,dim=-1)



        visual_feat_grd = torch.matmul(x_p, visual_feat_posi)              # cross modal attention
        visual_feat_grd_after_grounding_posi = visual_feat_grd.squeeze()    # [B*T, C]->[10B, 512]  

        visual_gl = torch.cat((visual_feat_before_grounding_posi, visual_feat_grd_after_grounding_posi),dim=-1) #[10B, 1024]
        visual_feat_grd = self.tanh(visual_gl)
        visual_feat_grd_posi = self.fc_gl(visual_feat_grd)              # [B*T, C]->[10B, 512]

       

        feat = torch.cat((audio_feat, visual_feat_grd_posi), dim=-1)    # [B*T, C*2], [B*T, 1024]

        feat = F.relu(self.fc1(feat))       # (1024, 512)
        feat = F.relu(self.fc2(feat))       # (512, 256)
        feat = F.relu(self.fc3(feat))       # (256, 128)
        out_match_posi = self.fc4(feat)     # (128, 2)

        ###############################################################################################
        # visual nega
        B, T, C, H, W = visual_nega.size()
        temp_visual = visual_nega.view(B*T, C, H, W)
        v_feat = self.avgpool(temp_visual)
        visual_feat_before_grounding_nega = v_feat.squeeze() # [B*T, C]

        (B, C, H, W) = temp_visual.size()
        v_feat = temp_visual.view(B, C, H * W)  # [B*T, C, HxW]
        v_feat = v_feat.permute(0, 2, 1)        # [B, HxW, C]
        visual_feat_nega = nn.functional.normalize(v_feat, dim=2)


        x1_it = torch.matmul(visual_feat_nega,q_text_tt).squeeze()

        x1_p = F.softmax(x1_it, dim=-1).unsqueeze(-2) 
       
      

        ##### av grounding nega
        x2_va = torch.matmul(visual_feat_nega, audio_feat_aa).squeeze()
        x2_p = F.softmax(x2_va, dim=-1).unsqueeze(-2)                       # [B*T, 1, HxW]
       
        x_p = x1_p+x2_p
        tvmask = x_p>0.01
        x_p = x_p*tvmask 
        x_p = F.softmax(x_p,dim=-1)

        visual_feat_grd = torch.matmul(x_p, visual_feat_nega)
        visual_feat_grd_after_grounding_nega = visual_feat_grd.squeeze()    # [B*T, C]   

        visual_gl=torch.cat((visual_feat_before_grounding_nega,visual_feat_grd_after_grounding_nega),dim=-1)
        visual_feat_grd=self.tanh(visual_gl)
        visual_feat_grd_nega=self.fc_gl(visual_feat_grd)    # [B*T, C]
 
       

        # combine a and v
        feat = torch.cat((audio_feat, visual_feat_grd_nega), dim=-1)   # [B*T, C*2], [B*T, 1024]
        
        
        feat = F.relu(self.fc1(feat))       # (1024, 512)
        feat = F.relu(self.fc2(feat))       # (512, 256)
        feat = F.relu(self.fc3(feat))       # (256, 128)
        out_match_nega = self.fc4(feat)     # (128, 2)

        ###############################################################################################

      

        B = xq.shape[1]
 


        visual_feat_grd_be = visual_feat_grd_posi.view(B, -1, 512)   # [B, T, 512]
        visual_feat_grd=visual_feat_grd_be.permute(1,0,2)            # [S(source sequence length), N, E]->[10, B, 512]
        
       
        # # attention, question as query on audio
        audio_feat_be=audio_feat_pure.view(B, -1, 512)   # [B, 10, 512]
        audio_feat = audio_feat_be.permute(1, 0, 2)
       
        
        self.a_gru.flatten_parameters()
        self.v_gru.flatten_parameters()
        gru_audio, a_hidden = self.a_gru(audio_feat_pure) #[B,10,512]
        gru_video, v_hidden = self.v_gru(visual_feat_grd_be) 
       
       
        video_feat = torch.cat([gru_video, gru_audio],dim=1) #[B,20,512]
       
      
        
        
        av_chunks = torch.chunk(video_feat,chunks=2,dim=1)
        video_feat = torch.stack([av_chunks[i % 2][:, i // 2, :] for i in range(2*audio_feat_pure.shape[1])], dim=1)

        video_feat = video_feat.permute(1,0,2) #[2T,B,C]
        


        avfeat_att, att_weights = self.attn_av(xq, video_feat, video_feat, attn_mask=None, key_padding_mask=None, need_weights=True)
        visual_feat_att = avfeat_att.squeeze(0)
        src = self.linearav2(self.dropoutav1(F.relu(self.linearav1(visual_feat_att))))  # [B, 512]
        visual_feat_att = visual_feat_att + self.dropoutav2(src)  # [B, 512]
        feat_av = self.normav(visual_feat_att)
        att_weights = att_weights.squeeze(1)
        vatt_weights = att_weights[:,::2]
        aatt_weights = att_weights[:,1::2]
        

        
      
        feat = torch.cat((audio_feat_be.mean(dim=-2).squeeze(), visual_feat_grd_be.mean(dim=-2).squeeze()), dim=-1)
        feat = self.tanh(feat)
        feat = self.fc_fusion(feat)
        
        feat = feat+feat_av
                      

        ## fusion with question
       
        combined_feature = torch.mul(feat, qst_feature)
        combined_feature = self.tanh(combined_feature)
        out_qa = self.fc_ans(combined_feature)              # [batch_size, ans_vocab_size]



        return out_qa, out_match_posi,out_match_nega, aatt_weights, vatt_weights
