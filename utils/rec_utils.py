import editdistance as ed
import numpy as np;
import random;
from similarity.longest_common_subsequence import LongestCommonSubsequence
class kot_ed_helper:
    @classmethod
    def ned(cls,a,b):
        return ed.eval(a,b)/(max(len(a),len(b))+0.000001);
    @classmethod
    def ned_matrix(cls,r,c):
        cnt_r=len(r);
        cnt_c=len(c);
        matrix=np.zeros((cnt_r,cnt_c));
        for r_id in range(cnt_r):
            for c_id in range(cnt_c):
                matrix[r_id,c_id]=cls.ned(r[r_id],c[c_id]);
        return matrix;
    # to what extend r[i] is a substring of c[j]?
    @classmethod
    def substring_score(cls,r,c,mask):
        lm=cls.get_masked_lcs_matrix(r,c,mask);
        ret=np.zeros_like(lm);
        for i in range(len(r)):
            for j in range(len(c)):
                if(mask[i,j]):
                    ret[i,j]=(lm[i,j]/len(r[i]+0.01))
        return ret;

    @classmethod
    def masked_ned_matrix(cls, r, c,mask):
        cnt_r = len(r);
        cnt_c = len(c);
        matrix = np.ones((cnt_r, cnt_c));
        for r_id in range(cnt_r):
            for c_id in range(cnt_c):
                if(mask[r_id,c_id]):
                    matrix[r_id, c_id] = cls.ned(r[r_id], c[c_id]);
        return matrix;
    @classmethod
    def ccs(cls,group,group_scores,candidates):
        cd={};

        th=np.sum(group_scores)/3;
        for i in range(len(group)):
            for k in set(group[i]):
                if k not in cd:
                    cd[k]=0;
                cd[k]+=group_scores[i];
        for k in cd.keys():
            cd[k]=(cd[k]>th);#/len(group);
        ss=[];
        for c in candidates:
            cs=[];

            if(len(c)):
                for cc in c:
                    cs.append(cd[cc]);
                ss.append(np.mean(cs));
            else:
                ss.append(0);

        ass=np.array(ss);
        return ass;

    @classmethod
    def cds(cls,group,group_scores,candidates):
        rsm=1-cls.ned_matrix(group,candidates);
        for i in range(len(group_scores)):
            rsm[i,:]*=group_scores[i];
        return np.array(np.mean(rsm,0));
    @classmethod
    def score(cls,group,group_scores,candidates):
        s1=cls.cds(group,group_scores,candidates);
        #s2=cls.ccs(group,group_scores,candidates);
        #return s1.tolist();
        return s1.tolist();

        #return ((10*s1+s2)/11).tolist();

    @classmethod
    def coinsistance(cls, group, group_scores, candidates):
        s1 = cls.cds(group, group_scores, candidates);
        return np.mean(s1);
    @classmethod
    def lcs_compact(cls,X,Y):
       return LongestCommonSubsequence.length(X,Y);

    @classmethod
    def lcs(cls,s1, s2):
        if(min(len(s1),len(s2))==0):
            return "";
        matrix = [["" for x in range(len(s2))] for x in range(len(s1))]
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    if i == 0 or j == 0:
                        matrix[i][j] = s1[i]
                    else:
                        matrix[i][j] = matrix[i - 1][j - 1] + s1[i]
                else:
                    matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1], key=len)

        cs = matrix[-1][-1];
        return cs
    # key must be a subsequence of res
    @classmethod
    def get_masked_lcs_matrix(cls,rows,cols,mask):
        lm=np.zeros((len(rows),len(cols)),float);

        for i in range(len(rows)):
            for j in range(len(cols)):
                if(mask[i,j]):
                    lm[i,j]=cls.lcs_compact(rows[i],cols[j]);
        return lm

    @classmethod
    def get_inc_patch(cls,key,res):
        sta=0;
        end=0;
        pss=[];
        if(len(key)==0):
            return [res];
        for i in range(len(key)):
            ps="";
            while(res[end] != key[i]):
                assert (end<len(res));
                ps+=res[end];
                end+=1;
            pss.append(ps);
            sta=end+1;
            end=sta;
        pss.append(res[sta:end]);
        return pss;

    @classmethod
    def patch(cls,key,patch):
        ret=patch[0];
        for i in range(len(key)):
            ret+=key[i];
            ret+=patch[i+1];
        return ret;

    @classmethod
    def indescribable_lcs(cls, a, b, a_s, b_s):
        key=cls.lcs(a,b);
        pa=cls.get_inc_patch(key,a);
        pb = cls.get_inc_patch(key, b);
        at=a_s/(a_s+b_s+0.000001);
        pc=[];
        for i in range(len(pa)):
            if(random.random()<at):
                pc.append(pa[i]);
            else:
                pc.append(pb[i]);
        return cls.patch(key,pc);

    @classmethod
    def indescribable_AB(cls, a, b, a_s, b_s):
        return a[:int(len(a)/2)]+b[int(len(b)/2):];

    @classmethod
    def indescribable_inter(cls, a, b, a_s, b_s):

        c="";
        al=len(a);
        bl=len(b);
        cl=int((al+bl)/2);
        acord=np.array(range(al))/al*cl;
        bcord = np.array(range(bl)) / bl*cl;
        ccord = np.array(range(cl));
        a_i=0;
        b_i=0;


        for c_i in range(cl):
            while(a_i+1<al and acord[a_i+1]<ccord[c_i]):
                a_i+=1;
            while (b_i+1 < bl and bcord[b_i + 1] < ccord[c_i]):
                b_i += 1;
            sels=[];
            weig=[];

            def tryput(sstr,sscore,scord,s_i):
                if(s_i>=len(scord)):
                    return ;
                try:
                    d = abs(scord[s_i] - ccord[c_i]);
                except:
                    pass;


                if (d > 0.95):
                    return ;
                score=1-d;
                if(score>0):
                    sels.append(sstr[s_i]);
                    weig.append(sscore*score)

            tryput(a, a_s, acord, a_i);
            tryput(a, a_s, acord, a_i+1);
            tryput(b, b_s, bcord, b_i);
            tryput(b, b_s, bcord, b_i+1);
            c+=random.choices(population=sels,weights=weig,k=1)[0];
        return c;

    @classmethod
    def evolve(cls,orig,orig_prescores,candidates,scores,topk):
        children=[];
        fns=[cls.indescribable_inter]
        for i in range(len(candidates)):
            for j in range(i+1,len(candidates)):
                indescribable=random.choice(fns);
                children.append(indescribable(candidates[i], candidates[j], scores[i], scores[j]));
        children=list(set(children))
        c_scores=cls.score(orig,orig_prescores,children);
        keep_idx=np.argsort(c_scores)[-topk:];
        ret_s=[];
        ret_c=[];
        topk=min(topk,len(children));

        for i in range(topk):
            ret_c.append(children[keep_idx[-i]]);
            ret_s.append(c_scores[keep_idx[-i]]);
        return ret_c,ret_s
    @classmethod
    def ensemble(cls,preds,pre_scores,_):
        scores=cls.score(preds,pre_scores,preds);

        #
        c_pred=[];
        c_scores=[];

        tpks=[]
        for i in tpks:
            c_pred+=preds;
            c_scores+=scores;
            c_pred,c_scores=cls.evolve(preds,pre_scores,c_pred,c_scores,i);

        fpreds=preds+c_pred;
        fscores=scores+c_scores;
        return fpreds[np.argmax(fscores)];

    @classmethod
    def ensemble_mk2(cls, preds, pre_scores,sups):
        scores = cls.score(preds, pre_scores, preds);
        sup_scores=cls.score(preds,prescores,sups);
        #
        c_pred = [];
        c_scores = [];

        tpks = []
        for i in tpks:
            c_pred += preds;
            c_scores += scores;
            c_pred, c_scores = cls.evolve(preds, pre_scores, c_pred, c_scores, i);

        fpreds = preds + c_pred +sups;
        fscores = scores + c_scores+sup_scores;

        return fpreds[np.argmax(fscores)];

def ensemble(keys,dicts,supdicts):
    res=[];
    tot=len(keys);
    idx=0;
    for k in keys:
        if(idx % 39 ==0):
            print(idx,"of",tot);
        idx+=1;
        props=[];
        scores=[];
        sups=[];
        for i in range(len( dicts)):
            d=dicts[i];
            try:
                props.append(d[k]);
                for sk in supdicts[k]:
                    sups.append(d[sk]);
                scores.append(prescores[i]);
            except:
                pass;
        res.append(kot_ed_helper.ensemble_mk2(props,scores,sups));
    return res;

from rec_eval.tencent_rec_eval import tre;
import os;
if __name__ == '__main__':
    files=os.listdir("/home/lasercat/cat/project_rnet_data/recognition/submits/");

    root="/home/lasercat/cat/project_rnet_data/recognition/submits/"
    dicts=[];
    rd={};
    # kot_ed_helper.score(['语', '美味', '羊', '盖'],['语', '美味', '羊', '盖']);
    # print(kot_ed_helper.indescribable_inter("abple","able",1,1));

    evaluator=tre("/home/lasercat/cat/project_rnet_data/recognition/gts/lsvt_gt.txt",
                  "/home/lasercat/cat/project_tf_family/rec_eval/STR_online_final_10514_simple.txt",
                  "/home/lasercat/cat/project_tf_family/rec_eval/STR_online_final_10514_traditional.txt"
                );
    for i in files:
        dicts.append(evaluator.get_dict(os.path.join(root,i)));
    keys=set();
    for d in dicts:
        keys=keys.union(set(d.keys()));
    kks={};
    for k in keys:
        kk=os.path.basename(k).split("_")[2];
        if(kk not in kks):
            kks[kk]=[];
        kks[kk].append(k);
    sup={};
    for k in keys:
        sup[k]=[];
        kk=kks[os.path.basename(k).split("_")[2]];
        for sk in kk:
            ned=[];
            for i in range(len(dicts)):
                ned.append(kot_ed_helper.ned(dicts[i][sk],dicts[i][k]));
            if(np.mean(ned)<0.4):
                sup[k].append(sk);


    prescores=[];
    for i in range(len(dicts)):
        prescores.append(1);
    from utils.libpy import n_even_chunks_naive;
    from functools import partial;
    from multiprocessing import Pool
    task_cnt = 9;
    pool = Pool(task_cnt);
    tks = n_even_chunks_naive(list(keys), task_cnt);
    tskf = partial(ensemble, dicts=dicts,supdicts=sup);
    ress=pool.map(tskf, tks);
    for i in range(len(tks)):
        for ki in range(len(tks[i])):
            rd[tks[i][ki]]=ress[i][ki];



    res=evaluator.eval_dict(rd);
    print(res);
