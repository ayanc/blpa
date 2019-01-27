#--Ayan Chakrabarti <ayan@wustl.edu>
import blpa.graph as gp

def Model(inSz,nCls,ng=18,avpool=True,qtype=0,WD=1e-4):
        
    g = gp.Graph(gp.SimpleData(inSz),gp.LogLoss())

    ch = [16,16,32,64]
    
    g.add(3,ch[0],1,'SAME',gp.NLDef(False,False,False,None,False))

    skip = gp.NLDef(True,True,False,None,False)
    join = gp.NLDef(True,True,True,1,False)
    
    g.add(1,ch[1],1,'SAME',gp.NLDef(True,True,False,2,False))
    g.add(3,ch[1],1,'SAME',skip)
    g.add(1,ch[1]*4,1,'SAME',skip)
    for i in range(ng-1):
        g.add(1,ch[1],1,'SAME',join)
        g.add(3,ch[1],1,'SAME',skip)
        g.add(1,ch[1]*4,1,'SAME',skip)
        
    g.add(1,ch[2],2,'SAME',join)
    g.add(3,ch[2],1,'SAME',skip)
    g.add(1,ch[2]*4,1,'SAME',skip)
    for i in range(ng-1):
        g.add(1,ch[2],1,'SAME',join)
        g.add(3,ch[2],1,'SAME',skip)
        g.add(1,ch[2]*4,1,'SAME',skip)

    g.add(1,ch[3],2,'SAME',join)
    g.add(3,ch[3],1,'SAME',skip)
    g.add(1,ch[3]*4,1,'SAME',skip)
    for i in range(ng-1):
        g.add(1,ch[3],1,'SAME',join)
        g.add(3,ch[3],1,'SAME',skip)
        g.add(1,ch[3]*4,1,'SAME',skip)

    fksz = 1 if avpool else inSz[1]//4    
    g.add(fksz,nCls,1,'VALID',gp.NLDef(True,True,True,None,avpool))

    g.close(qtype,WD)

    return g
