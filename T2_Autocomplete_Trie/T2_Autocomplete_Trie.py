
class TrieNode():
    ## Moi node bao gom tap children (dictionary)
    ## flag = True Noda la leaf.
    def __init__(self):
        self.children = {}
        self.flag = False

class Trie():
    ## Khoi tao root la node rong
    ## wordList luu cac chuoi tim duoc tu prefix
    def __init__(self):
        self.root = TrieNode()
        self.wordList = []

    ## Xay dung cay tu tap cac keys
    def createTrie(self,keys):
        for it in keys:
            self.addWord(it)


    def addWord(self, key):
        node = self.root
        for it in list(key):
            if not node.children.get(it):
                node.children[it] = TrieNode()

            node = node.children[it]
        node.flag = True


    ## Tra ve True neu : Chuoi da cho tao mot duong di
    ## tu goc den la
    def searchWord(self, word):
        node = self.root
        found = True
        for a in list(word):
            if not node.children.get(a):
                return False
                break
            node = node.children[a]
        return node and node.flag and found

    def suggestionString(self,node, word):
        if node.flag:
            self.wordList.append(word)
        for a,n in node.children.items():
            self.suggestionString(n, word + a)

    ## Dua ra cac chuoi bat bau boi chuoi input
    ## Luu cac chuoi tim duoc vao wordList
    def printsuggestString(self, word):
        node = self.root
        tempword = ''
        found = True

        for a in list(word):
            if not node.children.get(a):
                found = False
                break

            tempword += a
            node = node.children[a]

        if not found:
            return 0
        elif node.flag and not node.children :
            return -1

        self.suggestionString( node, tempword)
        for s in self.wordList:
            print(s)
        return 1


if __name__  == '__main__':

    keys = ["Gradient", "Gradient Descent", "GD", "Stochastic", "Stochasitc Gradient",
        "Stochastic Gradient Descent", "Adam", "Adam Optimization", "Adam Optimizer"]
    prefix = "Grd"

    t = Trie()

    t.createTrie(keys)

    found = t.printsuggestString(prefix)
    if(found <1):
        print('Khong tim duoc chuoi')







