# graph is implemented by a dictionary;
# keys are vertices and values are out- and in- edges of the corresponding vertex.
# vertices and edges are weighted. edges are directed as well.
class graph:
    
    # define vertex object
    class vertex:
        def __init__(self, vid=None, weight=None):
            self.vid = vid
            self.weight = weight
        
        def getID(self):
            return self.vid
        
        def getWeight(self):
            return self.weight
        
        def setID(self, vid):
            self.vid = vid
            
        def setWeight(self, weight):
            self.weight = weight
            
    # define edge object
    class edge:
        def __init__(self, edge=(None, None), weight=None):
            self.vrtx1 = edge[0]
            self.vrtx2 = edge[1]
            self.weight = weight
            
        def getEdge(self):
            return (self.vrtx1, self.vrtx2)
        
        def getWeight(self):
            return self.weight
        
        def setVertex1(self, vrtx1):
            self.vrtx1 = vrtx1
        
        def setVertex2(self, vrtx2):
            self.vrtx2 = vrtx2
        
        def setWeight(self, weight):
            self.weight = weight
            
             
    # graph constructor 
    def __init__(self,gdict=None):
        if gdict is None:
            gdict = {}
        self.gdict = gdict
    
    # return a list of vertices by their IDs
    def vertices(self):
        return [vertex.getID() for vertex in self.gdict.keys()]
    
    # return a list of edges
    def edges(self):
        return self.findedges()
    
    # return the whole dictionary
    def dictionary(self):
        cdict = {}
        for vertex in self.gdict.keys():
            outedges = [e.getEdge() for e in self.gdict[vertex]['out']]
            inedges = [e.getEdge() for e in self.gdict[vertex]['in']]
            cdict[vertex.getID()] = {'out': outedges, 'in': inedges}
        return cdict
    
    # import a whole dictionary
    def imprt(self, gdict):
        print('please implement me!')

    
    # return weight of a vertex
    def getVertexWeight(self, vrtx, show_msgs=True):
        for vertex in self.gdict:
            if vertex.getID() == vrtx:
                return vertex.getWeight()
        if show_msgs:
            print('vertex {} does not exist.'.format(vrtx))
            
    # return weight of an edge
    def getEdgeWeight(self, edge, show_msgs=True):
        (vrtx1, vrtx2) = edge
        for vertex in self.gdict:
            if vertex.getID() == vrtx1:
                for e in self.gdict[vertex]['out']:
                    if e.getEdge() == (vrtx1, vrtx2):
                        return e.getWeight()
        if show_msgs:
            print('edge {} does not exist'.format((vrtx1, vrtx2)))
    
    # edit a vertex weight (new weight is 0 if not specified)
    def setVertexWeight(self, vrtx, weight=None, show_msgs=True):
        if weight==None:
            weight = 0
            
        for vertex in self.gdict:
            if vertex.getID() == vrtx:
                vertex.setWeight(weight)
                return
        if show_msgs:
            print('vertex {} does not exist'.format(vrtx))
            
    # edit an edge weight (new weight is 1 if not specified)
    def setEdgeWeight(self, edge, weight=None, show_msgs=True):
        if weight==None:
            weight = 1
            
        (vrtx1, vrtx2) = edge
        for vertex in self.gdict:
            if vertex.getID() == vrtx1:
                for e in self.gdict[vertex]['out']:
                    if e.getEdge() == (vrtx1, vrtx2):
                        e.setWeight(weight)
                        return
        if show_msgs:
            print('edge {} does not exist'.format((vrtx1, vrtx2)))
    
    # Add new vertex (weight is 0 if not specified)
    def addVertex(self, vid=None, weight=None, show_msgs=True):
        if vid==None:
            vid=len(self.gdict)
            while vid in self.vertices():
                vid = vid + 1
                
        if weight==None:
            weight = 0
            
        if vid in self.vertices():
            if show_msgs:
                print('vertex id {} already exists'.format(vid))
        else:
            vrtx = self.vertex(vid=vid, weight=weight)
            self.gdict[vrtx] = {'out': [], 'in': []}
    
    # Add new directed edge (weight is 1 if not specificed)
    def addEdge(self, edge, weight=None, show_msgs=True):
        if weight==None:
            weight = 1
        (vrtx1, vrtx2) = edge
        
        if vrtx1 not in self.vertices():
            if show_msgs:
                print("vertex {} does not exist".format(vrtx1))     
        elif vrtx2 not in self.vertices():
            if show_msgs:
                print("vertex {} does not exist".format(vrtx2))  
        elif (vrtx1, vrtx2) in self.edges():
            if show_msgs:
                print("edge {} already exists".format((vrtx1,vrtx2)))
        else:
            e = self.edge(edge=(vrtx1, vrtx2), weight=weight)
            for vertex in self.gdict.keys():
                if vertex.getID() == vrtx1:
                    self.gdict[vertex]['out'].append(e)
                elif vertex.getID() == vrtx2:
                    self.gdict[vertex]['in'].append(e)
            
    # Add new undirected edge (weight is 1 if not specified)
    def addUndirectedEdge(self, edge, weight=None, show_msgs=True):
        if weight==None:
            weight = 1
        (vrtx1, vrtx2) = edge
        
        if vrtx1 not in self.vertices():
            if show_msgs:
                print("vertex {} does not exist".format(vrtx1))     
        elif vrtx2 not in self.vertices():
            if show_msgs:
                print("vertex {} does not exist".format(vrtx2))
        elif (vrtx1, vrtx2) in self.edges() and (vrtx2, vrtx1) in self.edges():
            if show_msgs:
                print("undirected edge {} already exists".format((vrtx1,vrtx2)))
        else:
            if (vrtx1, vrtx2) not in self.edges():
                e = self.edge(edge=(vrtx1, vrtx2), weight=weight)
                for vertex in self.gdict.keys():
                    if vertex.getID() == vrtx1:
                        self.gdict[vertex]['out'].append(e)
                    elif vertex.getID() == vrtx2:
                        self.gdict[vertex]['in'].append(e)
                        
            if (vrtx2, vrtx1) not in self.edges():
                e = self.edge(edge=(vrtx2, vrtx1), weight=weight)
                for vertex in self.gdict.keys():
                    if vertex.getID() == vrtx2:
                        self.gdict[vertex]['out'].append(e)
                    elif vertex.getID() == vrtx1:
                        self.gdict[vertex]['in'].append(e)
            

    # List the edge names
    def findedges(self):
        edgename = []
        for vertex in self.gdict:
            for e in self.gdict[vertex]['out']:
                edgename.append(e.getEdge())
        return edgename
