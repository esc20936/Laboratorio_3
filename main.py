from itertools import product

class BayesianNetwork:
    def __init__(self, prob_dict):
        self.prob_dict = prob_dict
        self.validate_prob_dict()
        self.compact_prob_dict()
        
    def validate_prob_dict(self):
        # check that sum of probabilities for each variable is 1
        for var, prob_info in self.prob_dict.items():
            prob_sum = sum(prob_info['probs'])
            if abs(prob_sum - 1) > 0.00001:
                raise ValueError(f"Invalid probability distribution for variable {var}: sum is not 1")
            if not all(isinstance(p, float) and 0 <= p <= 1 for p in prob_info['probs']):
                raise ValueError(f"Invalid probability distribution for variable {var}: probabilities must be between 0 and 1")
            if not all(isinstance(p, tuple) and len(p) == 2 and isinstance(p[0], str) and isinstance(p[1], str) for p in prob_info['parents']):
                raise ValueError(f"Invalid parent list for variable {var}")
            for parent in prob_info['parents']:
                if parent[0] not in self.prob_dict:
                    raise ValueError(f"Parent variable {parent[0]} not defined")
        
        # check for circular dependencies
        for var, prob_info in self.prob_dict.items():
            visited = set()
            for parent in prob_info['parents']:
                while parent[0] in self.prob_dict and parent[0] not in visited:
                    visited.add(parent[0])
                    parent = next((p for p in self.prob_dict[parent[0]]['parents'] if p[0] == var), None)
                    if parent is None:
                        break
                if parent is not None:
                    raise ValueError(f"Circular dependency between {var} and {parent[0]}")
                
    def compact_prob_dict(self):
        self.compact = ''
        for var, prob_info in self.prob_dict.items():
            if len(prob_info['parents']) == 0:
                self.compact += f"P({var})"
            else:
                parents = ''.join(p[0] for p in prob_info['parents'])
                self.compact += f"P({var}|{parents})"
        
    def inference(self, query):
        if query[0] not in self.prob_dict:
            raise ValueError(f"Invalid query variable: {query[0]}")
        prob_info = self.prob_dict[query[0]]
        parents = dict(prob_info['parents']) if prob_info['parents'] else {}
        observed = dict(query[1:])
        unobserved = set(parent[0] for parent in prob_info['parents']) - set(observed)
        prob = 0
        for vals in product(('0', '1'), repeat=len(unobserved)):
            cond_prob = 1
            for parent, parent_val in parents.items():
                if parent in observed:
                    prob_index = prob_info['parents'].index((parent, observed[parent]))
                else:
                    parent_val = int(vals[list(unobserved).index(parent)])
                    prob_index = prob_info['parents'].index((parent, str(parent_val)))
                cond_prob *= prob_info['probs'][prob_index]
            if all(observed.get(parent, str(parent_val)) == str(parent_val) for parent, parent_val in parents.items()):
                prob += cond_prob
        return prob / sum(prob_info['probs'])

    
    def get_compact_prob_dict(self):
        return self.compact
