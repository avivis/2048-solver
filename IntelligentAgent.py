#Avighna Suresh
#as6469

from BaseAI import BaseAI
import time

class IntelligentAgent(BaseAI):

    def getMove(self, grid):
        self.start_time = time.time()
        return self.ids(grid)

    def ids(self, g):
        bestmove = None
        tlimit = 0.2
        depth = 4
        while time.time()-self.start_time<tlimit and depth<=4:
            result = self.minimax(g, depth)
            if result is not None:
                if bestmove is None or result[0] > bestmove[0]:
                    bestmove = result
            depth += 1
        return bestmove[1]

    def minimax(self, g, depth, alpha=float('-inf'), beta=float('inf')):
        return self.maximize_fct(g, depth, alpha, beta)

    def maximize_fct(self, g, d, a, b):
        tlimit = 0.2
        if time.time()-self.start_time>=tlimit:
            return self.eval(g), None
        if d == 0:
            return self.eval(g), None
        if not g.canMove():
            return self.eval(g), None
        maxutil = float('-inf')
        maxchild = None
        avail_moves = g.getAvailableMoves()
        for m, c in avail_moves:
            util, _ = self.minimize_fct(c, d - 1, a, b)
            if util > maxutil:
                maxutil = util
                maxchild = m
            a = max(a, maxutil)
            if maxutil >= b:
                break
        return maxutil, maxchild

    def minimize_fct(self, g, d, a, b):
        tlimit = 0.2
        if time.time()-self.start_time>=tlimit:
            return self.eval(g), None
        if d == 0:
            return self.eval(g), None
        if not g.canMove():
            return self.eval(g), None
        minutil = float('inf')
        minchild = None
        for cell in g.getAvailableCells():
            child = g.clone()
            child.setCellValue(cell, 2)
            util, _ = self.maximize_fct(child, d - 1, a, b)
            child2 = g.clone()
            child2.setCellValue(cell, 4)
            util2, _ = self.maximize_fct(child2, d - 1, a, b) 
            finalutil=0 
            finalutil += 0.1 * util2 + 0.9 * util
            if finalutil < minutil:
                minutil = finalutil
                minchild = cell
            if finalutil >= b:
                a = finalutil
                minchild = cell
            if finalutil > a:
                break
        return minutil, minchild


    def eval(self, g):
        return (0.3 * len(g.getAvailableCells())) + (1.5 * self.snake(g.map)) + (0.5 * self.smoothness(g))

    def snake(self, g):
        maskval = []
        masks = [[[8, 4, 2, 1], [16, 32, 64, 128], [2048, 1024, 512, 256], [4096, 8192, 16384, 32768]], 
        [[1, 2, 4, 8], [128, 64, 32, 16], [256, 512, 1024, 2048], [32768, 16384, 8192, 4096]], 
        [[32768, 16384, 8192, 4096], [256, 512, 1024, 2048], [128, 64, 32, 16], [1, 2, 4, 8]], 
        [[4096, 8192, 16384, 32768], [2048, 1024, 512, 256], [16, 32, 64, 128], [8, 4, 2, 1]]]
        for mask in masks:
            evalmask = 0
            for i in range(4):
                for j in range(4):
                    evalmask += g[i][j] * mask[i][j]
            maskval.append(evalmask)
        return max(maskval)

    def smoothness(self, g):
        score=0
        for r in g.map:
            for i in range(len(r)-1):
                score -= abs(r[i]-r[i+1])
        for c in zip(*g.map):
            for i in range(len(c)-1):
                score-=abs(c[i]-c[i+1])
        return score

    def monotonicity(self, g):
        score = 0
        for r in g.map:
            for i in range(len(r)-1):
                if r[i] <= r[i+1]:
                    score+=r[i+1]-r[i]
        for c in zip(*g.map):
            for i in range(len(c)-1):
                if c[i] <= c[i+1]:
                    score += c[i+1]-c[i]
        return score