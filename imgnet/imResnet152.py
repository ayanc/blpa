#--Ayan Chakrabarti <ayan@wustl.edu>
import blpa.graph as gp
import blpa.mgraph as mgp
import imloader as dt

def Model(bsz,qtype=0,WD=1e-4,multi=False,ngpu=1):

    if ngpu == 1:
        g = gp.Graph(dt.IMData(bsz),gp.LogLoss())
    else:
        g = mgp.Graph([dt.IMData,[bsz]],[mgp.LogLoss,[]],ngpu)

    ch = [64,64,128,256,512]
    ng = [ 3,  8,  36,  3]
    
    g.add(7,ch[0],2,'SAME',gp.NLDef(False,False,False,None,False))

    skip = gp.NLDef(True,True,False,None,False)
    join = gp.NLDef(True,True,True,1,False)
    
    g.add(1,ch[1],1,'SAME',gp.NLDef(True,True,False,2,False,True))
    g.add(3,ch[1],1,'SAME',skip)
    g.add(1,ch[1]*4,1,'SAME',skip)
    for i in range(ng[0]-1):
        g.add(1,ch[1],1,'SAME',join)
        g.add(3,ch[1],1,'SAME',skip)
        g.add(1,ch[1]*4,1,'SAME',skip)

    for j in range(1,len(ng)):    
        g.add(1,ch[j+1],2,'SAME',join)
        g.add(3,ch[j+1],1,'SAME',skip)
        g.add(1,ch[j+1]*4,1,'SAME',skip)
        for i in range(ng[j]-1):
            g.add(1,ch[j+1],1,'SAME',join)
            g.add(3,ch[j+1],1,'SAME',skip)
            g.add(1,ch[j+1]*4,1,'SAME',skip)

    g.add(1,1000,1,'VALID',gp.NLDef(True,True,True,None,True))
    g.close(qtype,WD,multi)

    return g
