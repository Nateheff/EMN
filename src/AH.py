import torch
import torch.nn as nn
from dataclasses import dataclass

from modules.entorhinal import EntorhinalLayer
from modules.helpers import kWTA
from modules.compression import CompressionLayer
from modules.stage_1 import Stage1
from modules.stage_2 import Stage2
from modules.LTM import SAE_LTM
from modules.integration import IntegrationLayer    
from modules.output import OutputLayer

@dataclass
class AH_Args:
    proj_dim = 512
    embd_dim = 2048
    ent_dim = 1024
    expansion = 10
    k = int(ent_dim * expansion * 0.05) #512 for ~5% sparsity
    z_dim = ent_dim * expansion
    stage_1_hidden = 256
    stage_2_context_dim = proj_dim #Our context is coming from our stage 1
    num_latents = 8
    stage_2_emb = proj_dim
    num_layers = 2
    storage_dim = stage_2_emb
    retrieval_dim = stage_2_emb
    vae_dim = 512
    int_dim = ent_dim
    batch_size = 128
    mem_dim = 256
    



class AH(nn.Module):
    def __init__(self, args:AH_Args):
        super().__init__()
        self.args = args

        self.stage1_loss = nn.MSELoss()
        self.ltm_thresh = 0.7

        self.entorhinal = EntorhinalLayer(args.proj_dim, args.ent_dim)
        self.compression = CompressionLayer(args.ent_dim, args.expansion, args.k)
        self.stage_1 = Stage1(args.ent_dim, args.proj_dim, args.stage_1_hidden)
        self.stage_2 = Stage2(args.z_dim, args.stage_2_emb)
        self.ltm = SAE_LTM(args.storage_dim, args.retrieval_dim)
        self.integration = IntegrationLayer(args.mem_dim, args.ent_dim, args.vae_dim)
        self.output = OutputLayer(args.embd_dim, args.int_dim)

        self.ent_out  = None #Will need to store for Stage 1 update later
        self.query = None #Will need for feedback module


        self.sae_input = None
        self.sae_recon = None
        
        self.stage_2_buffer = []

        self.final_output = None

    def forward(self, query, captured_kv):

        ent_out = self.entorhinal(query)
        z_sparse = self.compression(ent_out)

        
        self.stage1_input = z_sparse

        high_recon, context_embed = self.stage_1(z_sparse, captured_kv['k'], captured_kv['v']) 
        recon_loss = self.stage1_loss(query, high_recon)

        while recon_loss >= self.recon_thresh:
            self.stage1_optim.zero_grad()
            recon_loss.backward()
            self.stage1_optim.step()

            high_recon, context_embed = self.stage_1(z_sparse, captured_kv['k'], captured_kv['v']) 
            recon_loss = self.stage1_loss(query, high_recon)


        self.stage2_input = { 'z_sparse': z_sparse, 'context': context_embed} #store in some buffer until offline consolidation
        self.stage_2_buffer.append(self.stage2_input)

        if len(self.stage_2_buffer) > 100:
            #TIME FOR OFFLINE CONSOLIDATION
            #loop through and concat z_sparse and context for each stored memory

            #run each through stage 2
            for input in self.stage_2_buffer:
                latents = self.stage_2(input['z_sparse'], input['context']) #[Batch, num_latents, latent_size]
                latent_full = torch.stack(latents)
                self.stage2_recon = latent_full
                self.sae_input = latent_full
                sae_recon, loss_sparse = self.ltm(latent_full, 0)
                self.sae_recon = sae_recon


    def store(self, query, captured_kv, consolidation=False):
        
        ent_out = self.entorhinal(query)
        fam_score = self.familiarity_store(ent_out)
        z_sparse = self.compression(ent_out)

        
        self.stage1_input = z_sparse
        

        while recon_loss >= self.recon_thresh:
            self.stage1_optim.zero_grad()
            recon_loss.backward()
            self.stage1_optim.step()

            high_recon, context_embed = self.stage_1(z_sparse, captured_kv['k'], captured_kv['v']) 
            recon_loss = self.stage1_loss(query, high_recon)


        self.stage2_input = { 'z_sparse': z_sparse, 'context': context_embed} #store in some buffer until offline consolidation
        self.stage_2_buffer.append(self.stage2_input)

        if consolidation:
            #TIME FOR OFFLINE CONSOLIDATION
            #loop through and concat z_sparse and context for each stored memory

            #run each through stage 2
            for input in self.stage_2_buffer:
                latents = self.stage_2(input['z_sparse'], input['context']) #[Batch, num_latents, latent_size]
                latent_full = torch.stack(latents)
                self.stage2_recon = latent_full
                self.sae_input = latent_full
                sae_recon, loss_sparse = self.ltm(latent_full, 0)
                self.sae_recon = sae_recon

    def retrieve(self, query, captured_kv):

        self.query = query
        
        ent_out = self.entorhinal(query)
        self.ent_out = ent_out

        mem_trace, use_ltm, ctx = self.stage_1(ent_out, captured_kv['k'], captured_kv['v'])

        z_sparse = self.compression(ent_out)
        abstract = self.stage_2(z_sparse, ctx) #[B, proj_dim]
         #[B, num_latents * latent_dim]
        sae_out, z_sparse_loss = self.ltm(abstract, 1)
    
        int_out = self.integration(mem_trace, ent_out, sae_out, use_ltm)
        final = self.output(int_out)
        self.final_output = final

        return final