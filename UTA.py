import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from mhar import walk
import torch


class UTA:
    def __init__(self, df, negative=list(), not_monotonic=list(), device='cpu', n_probes=150, n_points=-1,
                 single_goal=True):
        assert n_points < 0 or n_points > 1
        assert device in ['cpu', 'cuda']

        self.df = df
        self.device = device
        self.n_probes = n_probes
        self.better = []
        self.indifferent = []
        self.mappings = None
        self.mappingsNames = []
        self.x = None
        self.negative = negative
        self.not_monotonic = not_monotonic
        self.transformedDf = self.__transofrm(df)
        self.mappingsNames = list(self.transformedDf.columns)
        self.currentRes = None
        self.currentRes2 = None
        self.n_points = n_points
        self.single_goal = single_goal

    def __transofrm(self, x):
        result = x.copy()
        result[self.negative] = -result[self.negative]
        for k in self.not_monotonic:
            result[k + "-"] = -result[k]
        return result

    def __getXpFp(self, mapping, x, xx=None):
        if xx is None:
            xx = self.x
        i = self.mappingsNames.index(x)
        points = mapping[i]
        xp = []
        fp = []
        for q in points:
            xp += [q]
            fp += [xx[points[q]]]
        xp = np.array(xp)
        fp = np.array(fp)
        fp = fp[np.argsort(xp)]
        xp = xp[np.argsort(xp)]
        return xp, fp

    def __getUtilityFunctions(self, mapping=None, xx=None):
        xps = []
        fps = []
        if mapping is None:
            mapping = self.mappings
        if xx is None:
            xx = self.x
        for x in self.df.columns:
            if x not in self.not_monotonic:
                xp, fp = self.__getXpFp(mapping, x, xx)
                if x in self.negative:
                    xp = -xp[::-1]
                    fp = fp[::-1]
                xps.append(xp)
                fps.append(fp)
            else:
                xp, fp = self.__getXpFp(mapping, x, xx)
                xp2, fp2 = self.__getXpFp(mapping, x + "-", xx)
                xps.append(xp)
                fps.append(fp + fp2[::-1])
        return xps, fps

    def addBetter(self, x, y):
        self.better.append([x, y])
        self.__model("F")

    def getRanking(self):
        return self.__model("F")

    def getPrecision(self):
        return self.__model("T")
    
    def evaluate(self, mapping, x):
        return self.__evaluate(mapping, x)

    def __evaluate(self, mapping, x):
        result = np.zeros(self.df.shape[0])
        xps, fps = self.__getUtilityFunctions(mapping, x)
        for i in range(len(self.df.columns)):
            result += np.interp(self.df[self.df.columns[i]], xps[i], fps[i])
        return result

    def __getMappingVector(self, mapping, x, k):
        l = list(sorted([i for i in mapping]))
        first, size = min(mapping.values()), len(mapping)
        result = np.zeros(k)
        if x <= min(l):
            result[first] = 1
        if x >= max(l):
            result[first + size - 1] = 1
        for i in range(size - 1):
            if x > l[i + 1]: continue
            df = l[i + 1] - l[i]
            dx = l[i + 1] - x
            a = dx / df
            b = 1 - a
            result[first + i] = a
            result[first + i + 1] = b
            break
        return result

    def __model(self, mode="F"):
        used = []
        for x in self.better:
            used.append(x[0])
            used.append(x[1])
        used = self.__transofrm(pd.DataFrame(used, columns=self.df.columns))
        mappings = []
        mappingsNames = []
        i = 0
        for c in used.columns:
            tmp = {}
            if self.n_points < 0:
                valueSet = sorted(used[c].values)
            else:
                valueSet = np.linspace(self.transformedDf[c].min(), self.transformedDf[c].max(), self.n_points)
            for q in valueSet:
                if q not in tmp:
                    tmp[q] = i
                    i += 1
            mappings.append(tmp)
            mappingsNames.append(c)
        c = np.zeros(i + 1)
        a = np.zeros((len(self.better) + i - len(used.columns) + 2 * len(c) - 2, len(c)))
        b = np.zeros(len(self.better) + i - len(used.columns) + 2 * len(c) - 2)
        a_eq = np.zeros((2, len(c)))
        b_eq = np.zeros(2)
        for x in mappings:
            a_eq[0, min(x.values())] = 1
            a_eq[1, max(x.values())] = 1
        b_eq[1] = 1

        for i, (bet, wor) in enumerate(self.better):
            tmpDf = self.__transofrm(pd.DataFrame([bet, wor], columns=self.df.columns))
            for j, q in enumerate(tmpDf.iloc[0]):
                a[i] -= self.__getMappingVector(mappings[j], q, a.shape[1])
            for j, q in enumerate(tmpDf.iloc[1]):
                a[i] += self.__getMappingVector(mappings[j], q, a.shape[1])
        a[:len(self.better), -1] = 1
        i += 1
        bounds = [(0, 1) for x in c]
        bounds[-1] = (-1, None)

        for x in mappings:
            k = list(sorted(x.values()))
            for j in range(len(k) - 1):
                a[i][k[j]] = 1
                a[i][k[j + 1]] = -1
                i += 1

        for k in range(len(c) - 1):
            a[i][k] = -1
            b[i] = 0
            i += 1

        for k in range(len(c) - 1):
            a[i][k] = 1
            b[i] = 1
            i += 1

        if mode == "F":
            c[-1] = -1
            self.mappings = mappings
            self.x = linprog(c, a, b, a_eq, b_eq, bounds=bounds, method='revised simplex').x
            return self.__evaluate(mappings, self.x)
        else:
            a[:,-1] = 0
            result = []
            self.currentRes = []
            am = torch.as_tensor(a)
            am = am.type(torch.FloatTensor)

            bm = torch.as_tensor(b.reshape(-1, 1))
            bm = bm.type(torch.FloatTensor)

            aem = torch.as_tensor(a_eq)
            aem = aem.type(torch.FloatTensor)

            bem = torch.as_tensor(b_eq.reshape(-1, 1))
            bem = bem.type(torch.FloatTensor)

            res = linprog(c, a, b, bounds=bounds, )
            if res.status > 1:
                return -99
            x_0 = res.x
            x_0[-1] /= 2
            x_0 = torch.as_tensor(x_0.reshape(-1, 1))
            x_0 = x_0.type(torch.FloatTensor)
            X = walk(z=self.n_probes,
                     ai=am,
                     bi=bm,
#                      ae=aem,
#                      be=bem,
                     x_0=x_0,
                     T=1,
                     device=self.device,
                     warm=2,
                     seed=44,
                     thinning=15
                     ).numpy()
            
            for i in range(self.n_probes):
                if np.isnan(X[i]).any() or (X[i] > 1).any() or (X[i] < -1).any(): continue
                result.append(self.__evaluate(mappings, X[i]))
            self.currentRes = X
            self.currentRes2 = result
            if len(result) < 3:
                return -9
            return spearmanr(result, axis=1)[0].min()

    def __domination(self, x, y):
        used = [x, y]
        used = self.__transofrm(pd.DataFrame(used, columns=self.df.columns))
        return all(used.iloc[0] >= used.iloc[1]) or all(used.iloc[1] >= used.iloc[0])

    def scorePair(self, x, y):
        if self.__domination(x, y):
            print("domination")
            return -9
        self.better.append([x, y])
        result = self.__model("T")
        self.better = self.better[:-1] + [[y, x]]
        result = min(result, self.__model("T"))
        self.better = self.better[:-1]
        return result

    def plotUtilityFunctions(self):
        xps, yps = self.__getUtilityFunctions()
        for gj in range(len(self.df.columns)):
            gmin = self.df.iloc[:, gj].min()
            gmax = self.df.iloc[:, gj].max()
            maps = self.mappings[gj]
            xp = [gmin, *list(xps[gj]), gmax]
            yp = [yps[gj][0], *list(yps[gj]), yps[gj][-1]]
            print(xp)
            print(yp)
            print(xps[gj])
            print(yps[gj])
            plt.plot(xp, yp)
            plt.title(self.df.columns[gj])
            plt.show()