import torch.nn as nn
import torch.nn.functional as F
import torch
from .relational_table import RelationalTable
from .tag_net import TagNet
from .vidual_net import VidualNet


def distance(x, y, r):    
    x_norm_pow = x.pow(2).sum(dim=1).unsqueeze(dim=1)
    y_norm_pow = y.pow(2).sum(dim=1).unsqueeze(dim=1)
    r_norm_pow = r.pow(2).sum(dim=1).unsqueeze(dim=1)
    groval_notion = x.unsqueeze(dim=1).bmm(y.unsqueeze(dim=2)).squeeze(2)
    category_specific_notion = (y-x).unsqueeze(dim=1).bmm(r.unsqueeze(dim=2)).squeeze(2)

    out = x_norm_pow + y_norm_pow + r_norm_pow
    out -= 2*groval_notion + 2*category_specific_notion
    return out


def hinge_loss(d_near, d_far, margin=1.):
    loss = d_near - d_far + margin
    return torch.clamp(loss, min=0.)


class TransNFCM(nn.Module):
    def __init__(self, in_ch, out_ch,
                 n_relational_embeddings, n_tag_embeddings,
                 embedding_dim=128):
        super().__init__()
        self.vidual_net = VidualNet(in_ch, out_ch)
        self.relational_table = RelationalTable(n_relational_embeddings, embedding_dim*2)
        self.tag_net = TagNet(n_tag_embeddings, embedding_dim)
        self.embedding_dim = embedding_dim

    def calculate_VT_encoded_vec(self, v_vec, t_vec):
        v_vec = F.normalize(v_vec, p=2, dim=1)
        t_vec = F.normalize(t_vec, p=2, dim=1)
        vt_vec = torch.cat([v_vec, v_vec], dim=1)
        return vt_vec

    def calculate_distance(self, x, y, x_cat, y_cat, relational_tag):
        x_vidual_vec = self.vidual_net(x)
        y_vidual_vec = self.vidual_net(y)
        x_tag_vec = self.tag_net(x_cat)
        y_tag_vec = self.tag_net(y_cat)
        relational_vec = self.relational_table(relational_tag)
        relational_vec = F.normalize(relational_vec, p=2, dim=1).squeeze(dim=1)

        x_vt = self.calculate_VT_encoded_vec(x_vidual_vec, x_tag_vec)
        y_vt = self.calculate_VT_encoded_vec(y_vidual_vec, y_tag_vec)
        return distance(x_vt, y_vt, relational_vec)

    def predict(self, x=None, category=None):
        if x is None and category is not None:
            return F.normalize(self.tag_net(category), p=2, dim=1)
        elif x is not None and category is None:
            return F.normalize(self.vidual_net(x), p=2, dim=1)
        elif x is not None and category is not None:
            vidual_vec = self.vidual_net(x)
            tag_vec = self.tag_net(category)
            vt = self.calculate_VT_encoded_vec(vidual_vec, tag_vec)
            return F.normalize(vt, p=2, dim=1)
        else:
            assert False, "image or category, either one is required"

    def forward(self,
                x_near, y_near, x_far, y_far,
                x_near_cat, y_near_cat, x_far_cat, y_far_cat,
                near_relational_tag, far_relational_tag):
        near_dist = self.calculate_distance(x_near, y_near,
                                            x_near_cat, y_near_cat,
                                            near_relational_tag)
        far_dist =  self.calculate_distance(x_far, y_far,
                                            x_far_cat, y_far_cat,
                                            far_relational_tag)
        return hinge_loss(near_dist, far_dist)