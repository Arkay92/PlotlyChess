"""Chess adaptation of v18: categorical product circuits and a neural soft gate.

Classes are loss/draw/win for the side to move. Experts are bootstrapped
generative circuits; the gate sees posterior evidence, never outcome labels.
"""
import chess
import numpy as np

VALUES = {chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3, chess.ROOK:5, chess.QUEEN:9, chess.KING:0}
CARDINALITIES = (9, 3, 3, 3, 3)

def concepts(board):
    side = board.turn
    material = sum(v*(len(board.pieces(p,side))-len(board.pieces(p,not side))) for p,v in VALUES.items())
    return np.array([int(np.clip(material,-4,4))+4, min(2,len(list(board.legal_moves))//15),
                     int(board.is_check()), min(2,len(board.piece_map())//12),
                     min(2,sum(board.is_capture(m) for m in board.legal_moves))])

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e/e.sum(axis=-1, keepdims=True)

def gate_features(p):
    mean = p.mean(axis=1,keepdims=True)
    ordered = np.sort(p,axis=-1)
    return np.concatenate([-(p*np.log(p+1e-12)).sum(-1), ordered[:,:,-1]-ordered[:,:,-2],
                           p[:,:,2], p[:,:,0], (p*np.log((p+1e-12)/(mean+1e-12))).sum(-1)],axis=1)

class CircuitRouter:
    """Normalized sum/product circuits plus a tanh neural routing network."""
    def __init__(self, prior, leaves, w1, b1, w2, b2, mean, scale):
        self.prior, self.leaves = prior, leaves
        self.w1,self.b1,self.w2,self.b2 = w1,b1,w2,b2
        self.mean,self.scale = mean,scale

    def experts(self, x):
        logp = np.broadcast_to(np.log(self.prior), (len(x),*self.prior.shape)).copy()
        for j,leaf in enumerate(self.leaves):
            logp += np.log(leaf[:,:,x[:,j]].transpose(2,0,1))
        return softmax(logp)

    def predict(self, x):
        p = self.experts(np.asarray(x))
        features = (gate_features(p)-self.mean)/self.scale
        weights = softmax(np.tanh(features@self.w1+self.b1)@self.w2+self.b2)
        return (p*weights[:,:,None]).sum(1), weights

    def save(self, path):
        np.savez_compressed(path, prior=self.prior, w1=self.w1,b1=self.b1,w2=self.w2,b2=self.b2,
                            mean=self.mean,scale=self.scale, **{f"leaf{j}":v for j,v in enumerate(self.leaves)})

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            return cls(d['prior'],[d[f'leaf{j}'] for j in range(len(CARDINALITIES))],
                       *(d[k] for k in ('w1','b1','w2','b2','mean','scale')))

    @classmethod
    def fit(cls, x, y, gate_x, gate_y, seed=42, epochs=300, progress=None):
        """Fit experts on training games, gate on separate games. No test data."""
        rng = np.random.default_rng(seed)
        prior = np.ones((3,3))
        leaves = [np.ones((3,3,k)) for k in CARDINALITIES]
        if progress:
            progress('Experts', 0, 3*len(x), None)
        for expert in range(3):
            indices = rng.choice(len(x),len(x),replace=True)
            for count,(row,label) in enumerate(zip(x[indices],y[indices]),1):
                prior[expert,label] += 1
                for j,value in enumerate(row):
                    leaves[j][expert,label,value] += 1
                if progress and (count % 10000 == 0 or count == len(x)):
                    progress('Experts', expert*len(x)+count, 3*len(x), None)
        prior /= prior.sum(-1,keepdims=True)
        leaves = [v/v.sum(-1,keepdims=True) for v in leaves]
        model = cls(prior,leaves,rng.normal(0,.1,(15,16)),np.zeros(16),rng.normal(0,.1,(16,3)),np.zeros(3),0,1)
        p = model.experts(gate_x)
        f = gate_features(p)
        model.mean,model.scale = f.mean(0),np.maximum(f.std(0),1e-6)
        f = (f-model.mean)/model.scale
        true_p = p[np.arange(len(p)),:,gate_y]
        for epoch in range(epochs):
            h = np.tanh(f@model.w1+model.b1)
            w = softmax(h@model.w2+model.b2)
            mixture = (w*true_p).sum(1,keepdims=True)
            grad = w*(1-true_p/np.maximum(mixture,1e-12))/len(f)
            dh = (grad@model.w2.T)*(1-h*h)
            model.w2 -= .05*h.T@grad
            model.b2 -= .05*grad.sum(0)
            model.w1 -= .05*f.T@dh
            model.b1 -= .05*dh.sum(0)
            if progress and (epoch == 0 or (epoch+1) % 10 == 0 or epoch+1 == epochs):
                progress('Neural gate', epoch+1, epochs, float(-np.log(np.maximum(mixture,1e-12)).mean()))
        return model

class ChessAI:
    def __init__(self, model_path=None):
        self.router = CircuitRouter.load(model_path) if model_path else None

    @property
    def name(self):
        return "NPC conditional routing" if self.router else "Material search (train an NPC model to enable routing)"

    def evaluate(self, board, side):
        outcome = board.outcome(claim_draw=True)
        if outcome:
            return 0 if outcome.winner is None else (10000 if outcome.winner == side else -10000)
        return sum(v*(len(board.pieces(p,side))-len(board.pieces(p,not side))) for p,v in VALUES.items())

    def choose(self, board):
        if board.is_game_over(claim_draw=True):
            raise ValueError("No move: game is over.")
        moves = list(board.legal_moves)
        scores, rows = [], []
        side = board.turn
        for move in moves:
            child = board.copy()
            child.push(move)
            rows.append(concepts(child))
            replies = []
            for reply in child.legal_moves:
                following = child.copy()
                following.push(reply)
                replies.append(self.evaluate(following,side))
            scores.append(self.evaluate(child,side) if child.is_game_over(claim_draw=True) else min(replies))
        if self.router:
            p, weights = self.router.predict(np.asarray(rows))
            # Children have the opponent to move: their loss is our win.
            scores = np.asarray(scores) + .5*(p[:,0]-p[:,2])
            chosen = int(np.argmax(scores))
            explanation = "Expert weights: " + ", ".join(f"{w:.0%}" for w in weights[chosen])
        else:
            chosen = int(np.argmax(scores))
            explanation = "Two-ply material search."
        return moves[chosen], explanation
