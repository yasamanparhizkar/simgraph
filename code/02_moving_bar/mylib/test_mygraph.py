from mylib.mygraph import graph

# test adding vertices to the graph
def test_graph_1(show_msgs=True):
    g = graph()
    assert g.vertices() == []
    assert g.edges() == []
    assert g.dictionary() == {}
    
    g.addVertex()
    assert g.getVertexWeight(vrtx=0) == 0
    assert g.vertices() == [0]
    
    g.addVertex()
    g.addVertex(weight=1.5)
    g.addVertex(vid='2', weight=3)
    g.addVertex(weight='4')
    assert g.vertices() == [0, 1, 2, '2', 4]
    assert g.getVertexWeight(vrtx=2) == 1.5
    assert g.getVertexWeight(vrtx='2') == 3
    assert g.getVertexWeight(vrtx=4) == '4'
    
    g.getVertexWeight(vrtx=6, show_msgs=show_msgs)
    
    g.addVertex(vid=6)
    g.addVertex()
    assert g.vertices() == [0, 1, 2, '2', 4, 6, 7]
    
    g.addVertex(vid=7, show_msgs=False)
    assert g.vertices() == [0, 1, 2, '2', 4, 6, 7]
    
# test adding edges to the graph
def test_graph_2(show_msgs=True):
    g = graph()
    assert g.edges() == []
    
    for i in range(5):
        g.addVertex()
    g.addEdge((0,1))
    g.addEdge((1,0), weight=2)
    g.addEdge((2,3), weight=5)
    assert g.vertices() == [0, 1, 2, 3, 4]
    assert g.edges() == [(0, 1), (1, 0), (2, 3)]
    assert g.dictionary()[0] == {'out': [(0, 1)], 'in': [(1, 0)]}
    assert g.dictionary()[1] == {'out': [(1, 0)], 'in': [(0, 1)]}
    assert g.dictionary()[2] == {'out': [(2, 3)], 'in': []}
    assert g.dictionary()[3] == {'out': [], 'in': [(2, 3)]}
    assert g.dictionary()[4] == {'out': [], 'in': []}
    
    assert g.getEdgeWeight((0, 1)) == 1
    assert g.getEdgeWeight((1,0)) == 2
    assert g.getEdgeWeight((2,3)) == 5
    
    g.getEdgeWeight((3,2), show_msgs=show_msgs)
    
    g.addEdge((0,5), show_msgs=show_msgs)
    g.addEdge((6,1), show_msgs=show_msgs)
    g.addEdge((2,3), show_msgs=show_msgs)
    assert g.getEdgeWeight((1,3), show_msgs=show_msgs) == None
    
    g.addUndirectedEdge((4,3),weight=9)
    g.addUndirectedEdge((2,3),weight=5)
    g.addEdge((1,4))
    g.addUndirectedEdge((4,1),weight=3)
    g.addUndirectedEdge((1,0),weight=9, show_msgs=show_msgs)
    assert g.edges() == [(0, 1), (1, 0), (1, 4), (2, 3), (3, 4), (3, 2), (4, 3), (4, 1)]
    
    
# test editting vertex and edge weights
def test_graph_3(show_msgs=True):
    g = graph()
    for i in range(5):
        g.addVertex()
    for i in range(5):
        g.addUndirectedEdge((i,(i+1)%5))
    assert g.vertices() == [0, 1, 2, 3, 4]
    assert g.edges() == [(0, 1), (0, 4), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 0)]
    for i in range(5):
        assert g.getVertexWeight(vrtx=i) == 0
        assert g.getEdgeWeight(edge=(i,(i+1)%5)) == 1
        assert g.getEdgeWeight(edge=((i+1)%5,i)) == 1
        
    g.setVertexWeight(vrtx=2, weight=2)
    g.setVertexWeight(vrtx=1, weight=3)
    g.setVertexWeight(vrtx=7, weight=4, show_msgs=show_msgs)
    g.setEdgeWeight(edge=(0,1), weight=3)
    g.setEdgeWeight(edge=(2,3), weight='a')
    g.setEdgeWeight(edge=(1,3), weight=3, show_msgs=show_msgs)
    assert g.getVertexWeight(vrtx=0) == 0
    assert g.getVertexWeight(vrtx=1) == 3
    assert g.getVertexWeight(vrtx=2) == 2
    assert g.getVertexWeight(vrtx=3) == 0
    assert g.getVertexWeight(vrtx=4) == 0
    assert g.getEdgeWeight(edge=(0,1)) == 3
    assert g.getEdgeWeight(edge=(1,0)) == 1
    assert g.getEdgeWeight(edge=(1,2)) == 1
    assert g.getEdgeWeight(edge=(2,1)) == 1
    assert g.getEdgeWeight(edge=(2,3)) == 'a'
    assert g.getEdgeWeight(edge=(3,2)) == 1
    assert g.getEdgeWeight(edge=(3,4)) == 1
    assert g.getEdgeWeight(edge=(4,3)) == 1
    assert g.getEdgeWeight(edge=(4,0)) == 1
    assert g.getEdgeWeight(edge=(0,4)) == 1
        
    
def test_graph(show_msgs=True):      
    test_graph_1(show_msgs=show_msgs) # adding vertices to the graph
    test_graph_2(show_msgs=show_msgs) # adding edges to the graph
    test_graph_3(show_msgs=show_msgs) # editting vertex and edge weights
    
test_graph(show_msgs=False)
