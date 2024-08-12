import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stop
from .transformer_layers import SelfAttnLayer
from .backbone import Backbone, Backbone1, Backbone2, Backbone6
from .utils import custom_replace,weights_init
from .position_enc import PositionEmbeddingSine,positionalencoding2d


class CTranModel(nn.Module):
    # Original
    # def __init__(self,num_labels,use_lmt,pos_emb=False,layers=3,heads=4,dropout=0.1,int_loss=0,no_x_features=False):
    
    # Modified before 26/3/2024. Content: train 100.
    #def __init__(self, train_obs, val_obs, num_labels,use_lmt,pos_emb=False,layers=3,heads=4,dropout=0.1,int_loss=0,no_x_features=False):


    # Modified after 26/3/2024. Content: train 80, valid 20. 
    def __init__(self, train_obs, val_obs, test_obs, num_labels,use_lmt,pos_emb=False,layers=3,heads=4,dropout=0.1,int_loss=0,no_x_features=False):   
        super(CTranModel, self).__init__()
        self.use_lmt = use_lmt
        
        self.no_x_features = no_x_features # (for no image features)

        # ResNet backbone
        # self.backbone = Backbone()

        # Efficientnet V0 backbone
        # self.backbone = Backbone1()

        # Mobilenet V2 backbone
        self.backbone = Backbone6()
        
        # Backbone 2
        # self.backbone = Backbone2(train_obs, val_obs) # train 100


        hidden = 2048 # this should match the backbone output feature size

        self.downsample = False
        if self.downsample:
            self.conv_downsample = torch.nn.Conv2d(hidden,hidden,(1,1))
        
        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1,-1).long()
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None)

        # State Embeddings
        self.known_label_lt = torch.nn.Embedding(3, hidden, padding_idx=0)

        # Original version
        # Position Embeddings (for image features)
        # self.use_pos_enc = pos_emb
        # if self.use_pos_enc:
        #     # self.position_encoding = PositionEmbeddingSine(int(hidden/2), normalize=True)
        #     self.position_encoding = positionalencoding2d(hidden, 18, 18).unsqueeze(0)

        # Updated version
        self.use_pos_enc = False
        if self.use_pos_enc:
            # self.position_encoding = PositionEmbeddingSine(int(hidden/2), normalize=True)
            # self.position_encoding = positionalencoding2d(hidden, 18, 18).unsqueeze(0)
            self.position_encoding = positionalencoding2d(hidden, 18, 18)

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden,heads,dropout) for _ in range(layers)])

        # Classifier
        # Output is of size num_labels because we want a separate classifier for each label
        self.output_linear = torch.nn.Linear(hidden,num_labels)

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)


    def forward(self,images,mask):

        const_label_input = self.label_input.repeat(images.size(0),1)
        if torch.cuda.is_available():
            const_label_input = const_label_input.cuda()

        init_label_embeddings = self.label_lt(const_label_input)

        # If not use backbone 2
        features = self.backbone(images)

        # If use backbone 2
        # features = self.backbone.f(images)
        
        if self.downsample:
            features = self.conv_downsample(features)
        if self.use_pos_enc:
            # Original
            # pos_encoding = self.position_encoding(features,torch.zeros(features.size(0),18,18, dtype=torch.bool).cuda())
            # features = features + pos_encoding

            # Updated version
            # zeros_tensor = torch.zeros(size=(features.size(0), 18, 18), dtype=torch.bool).cuda()

            pos_encoding = self.position_encoding
            if torch.cuda.is_available():
                pos_encoding = pos_encoding.cuda()

            features = features + pos_encoding

        features = features.view(features.size(0),features.size(1),-1).permute(0,2,1)


        if self.use_lmt:
            # Convert mask values to positive integers for nn.Embedding
            label_feat_vec = custom_replace(mask,0,1,2).long()

            # Get state embeddings
            state_embeddings = self.known_label_lt(label_feat_vec)
            
            # Add state embeddings to label embeddings
            init_label_embeddings += state_embeddings

            # Modified on 3/3/2024. Content: add state embeddings to label embeddings.
            # init_label_embeddings *= state_embeddings
          
        if self.no_x_features:
            embeddings = init_label_embeddings 
        else:
            # Concat image and label embeddings
            embeddings = torch.cat((features,init_label_embeddings),1)

        # Feed image and label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)

 
        attns = []
        for layer in self.self_attn_layers:
            embeddings,attn = layer(embeddings,mask=None)
            attns += attn.detach().unsqueeze(0).data

        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]

        output = self.output_linear(label_embeddings) 

        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0),1,1)
        if torch.cuda.is_available():
            diag_mask = diag_mask.cuda()

        output = (output*diag_mask).sum(-1)
    
        return output,None,attns

