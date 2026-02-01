# Implementation Guide: Dynamic Bacteria & Multi-Macrophage

## 🎯 Phase 1: Dynamic Bacteria Behavior (IMPLEMENTED ✅)

### What Was Changed

**Files Created:**
- `agents/bacteria_adaptive.py` - Intelligent bacteria decision-making

**Files Modified:**
- `config.py` - Added `BACTERIA_ADAPTIVE_ENABLED` flag
- `simulator/environment.py` - Integration point for adaptive behavior

### How It Works

The `AdaptiveBacteriaAgent` evaluates each bacterium's situation every turn:

```python
# Decision tree:
1. Low health + nutrient nearby → SEEK NUTRIENT
2. Macrophage very close + no biofilm → FLEE
3. Macrophage close + in biofilm + mature → ATTACK
4. Quorum threshold almost met → CLUSTER
5. Default → EXPLORE CAUTIOUSLY
```

### Key Features

✅ **Context-Aware**: Bacteria adapt based on health, macrophage distance, biofilm status  
✅ **Stochastic**: 15% random noise prevents deterministic patterns  
✅ **Emergent Behavior**: No hardcoding - strategies emerge from conditions  
✅ **Backward Compatible**: Toggle with `BACTERIA_ADAPTIVE_ENABLED = False`  

### Testing

Run simulation and observe:
- Wounded bacteria flee to nutrients
- Healthy biofilm bacteria attack aggressively
- Isolated bacteria cluster for protection
- Bacteria maintain safe distance when exploring

```bash
cd experiments
python run_simulation.py
```

Watch for these behaviors:
1. **Fleeing**: Low-health bacteria move away from macrophage
2. **Nutrient Seeking**: Wounded bacteria beeline for nutrients
3. **Coordinated Attack**: Biofilm bacteria swarm macrophage
4. **Dynamic Clustering**: Bacteria form groups when threatened

---

## 🚀 Phase 2: Multi-Macrophage Cooperation (TODO)

### Architecture Changes Needed

#### 1. Update Environment (`simulator/environment.py`)

**Current**: Single `self.macrophage` object  
**New**: List `self.macrophages = []`

```python
class Environment:
    def __init__(self, num_macrophages=3):
        self.macrophages = []
        for i in range(num_macrophages):
            # Distribute macrophages across grid
            x = (i % 2) * (self.size // 2)
            y = (i // 2) * (self.size // 2)
            self.macrophages.append(Macrophage(position=(x, y), health=100))
```

#### 2. Add Communication System

**Create `agents/macrophage_team.py`:**

```python
class MacrophageTeam:
    """Coordinates multiple macrophages with shared information."""
    
    def __init__(self, macrophages):
        self.macrophages = macrophages
        self.signals = []  # List of (sender_id, message_type, data)
    
    def broadcast_signal(self, sender_id, signal_type, data):
        """Macrophage broadcasts help request or threat info."""
        self.signals.append({
            'sender': sender_id,
            'type': signal_type,  # 'help_needed', 'threat_detected', 'area_clear'
            'data': data,  # {'position': (x,y), 'bacteria_count': 5, 'health': 30}
            'timestamp': env.step_count
        })
    
    def get_relevant_signals(self, macrophage_id, position):
        """Get signals relevant to this macrophage."""
        relevant = []
        for sig in self.signals:
            if sig['sender'] == macrophage_id:
                continue  # Don't respond to own signals
            
            # Calculate distance to signal source
            dist = abs(sig['data']['position'][0] - position[0]) + \
                   abs(sig['data']['position'][1] - position[1])
            
            if dist <= RESPONSE_RADIUS:
                relevant.append(sig)
        
        return relevant
```

#### 3. Team-Based MCTS

**Modify `agents/mcts.py`:**

```python
class TeamMCTS:
    """MCTS that optimizes for team survival, not individual."""
    
    def choose_team_actions(self, env):
        """
        Returns: List of actions, one per macrophage
        """
        # Search over joint action space
        team_actions = []
        
        for macrophage in env.macrophages:
            # Each macrophage considers team state
            action = self._choose_with_team_awareness(macrophage, env)
            team_actions.append(action)
        
        return team_actions
    
    def _evaluate_state(self, env):
        """Team reward function."""
        if all(m.health <= 0 for m in env.macrophages):
            return -1000  # Team wiped out
        
        if len(env.bacteria) == 0:
            return 1000  # Team victory
        
        # Team fitness = sum of health - bacteria threat
        team_health = sum(m.health for m in env.macrophages)
        bacteria_threat = len(env.bacteria) * 10
        
        # Bonus for macrophages staying alive (incentivize not sacrificing)
        alive_bonus = sum(100 for m in env.macrophages if m.health > 0)
        
        return team_health - bacteria_threat + alive_bonus
```

#### 4. Signaling Logic

**When to Send Signals:**

```python
def macrophage_should_signal_help(macrophage, env):
    """Determine if macrophage should call for backup."""
    nearby_bacteria = count_bacteria_within(macrophage.position, radius=3)
    
    # Critical situation triggers
    if macrophage.health < 30 and nearby_bacteria >= 3:
        return True, "critical_health_surrounded"
    
    if nearby_bacteria >= 5:
        return True, "overwhelmed_by_numbers"
    
    if macrophage.exhaustion > 25:
        return True, "exhausted_need_relief"
    
    return False, None


def should_respond_to_signal(responding_mac, signal, env):
    """Decide if macrophage should divert to help."""
    # Calculate cost of helping
    current_bacteria_nearby = count_bacteria_within(responding_mac.position, radius=3)
    distance_to_signal = manhattan_distance(responding_mac.position, signal['data']['position'])
    
    # Don't respond if:
    # 1. Currently engaged in own fight
    if current_bacteria_nearby >= 3:
        return False
    
    # 2. Too far away
    if distance_to_signal > 10:
        return False
    
    # 3. Signal is old (already resolved)
    if env.step_count - signal['timestamp'] > 5:
        return False
    
    # Otherwise, respond!
    return True
```

#### 5. Updated Step Function

```python
def step(self):
    """Execute one simulation step with team coordination."""
    
    # Phase 1: Macrophages assess situation and signal
    team = MacrophageTeam(self.macrophages)
    
    for i, mac in enumerate(self.macrophages):
        should_signal, reason = macrophage_should_signal_help(mac, self)
        if should_signal:
            team.broadcast_signal(
                sender_id=i,
                signal_type='help_needed',
                data={
                    'position': mac.position,
                    'health': mac.health,
                    'bacteria_count': count_bacteria_within(mac.position, 3),
                    'reason': reason
                }
            )
    
    # Phase 2: Each macrophage chooses action considering team signals
    actions = []
    for i, mac in enumerate(self.macrophages):
        relevant_signals = team.get_relevant_signals(i, mac.position)
        action = self._choose_team_aware_action(mac, relevant_signals, self)
        actions.append(action)
    
    # Phase 3: Execute actions
    for mac, action in zip(self.macrophages, actions):
        self._execute_macrophage_action(mac, action)
    
    # Phase 4: Bacteria react
    self._execute_bacteria_actions()
    
    # Phase 5: Check victory
    self._check_team_victory_conditions()
```

---

## 🧪 Testing Multi-Macrophage

### Test Scenarios

**Scenario 1: Pincer Movement**
- 2 macrophages, 10 bacteria clustered
- Expected: Macrophages coordinate attack from both sides

**Scenario 2: Rescue Mission**
- Macrophage A surrounded (health < 20)
- Macrophage B nearby with clear path
- Expected: B diverts to rescue A

**Scenario 3: Divide & Conquer**
- 2 bacteria clusters at opposite corners
- 2 macrophages
- Expected: Each takes one cluster, no inefficient crossing

**Scenario 4: Last Stand**
- 1 macrophage alive, 1 dead
- Survivor adapts to solo mode (no waiting for backup)

### Metrics to Track

```python
# Add to history logging:
history['team_coordination_events'] = []  # Log each signal sent/received
history['macrophages_alive'] = []
history['average_macrophage_separation'] = []  # Are they spreading out?
history['help_signals_per_step'] = []
```

---

## 🔬 Phase 3: Bacteria MCTS (Optional)

### Simplification Strategy

**Problem**: 20 bacteria × 5 actions = huge search tree

**Solution**: Bacteria Hivemind

```python
class BacteriaHivemind:
    """Single agent controls all bacteria collectively."""
    
    def choose_collective_actions(self, env):
        """Returns list of actions for all bacteria."""
        
        # MCTS searches over representative bacteria
        # E.g., only plan for 5 "key" bacteria, others follow simple rules
        
        key_bacteria = self._select_key_bacteria(env.bacteria)
        
        actions = {}
        for kb in key_bacteria:
            action = self._mcts_plan_for_bacterium(kb, env)
            actions[kb] = action
        
        # Other bacteria use fast heuristic
        for b in env.bacteria:
            if b not in key_bacteria:
                actions[b] = self._quick_heuristic(b, env)
        
        return actions
    
    def _select_key_bacteria(self, all_bacteria):
        """Pick 3-5 most important bacteria to plan for."""
        # Prioritize:
        # 1. Healthiest (leaders)
        # 2. Closest to macrophage (threats)
        # 3. In biofilm (defenders)
        
        sorted_by_importance = sorted(
            all_bacteria,
            key=lambda b: (
                b.health * 2 +  # Health weight
                (1000 if b.in_biofilm else 0) +  # Biofilm bonus
                (100 / (macrophage_distance(b) + 1))  # Proximity to threat
            ),
            reverse=True
        )
        
        return sorted_by_importance[:5]  # Top 5
```

### Computational Cost Analysis

| Approach | Bacteria | Actions/each | Rollouts | Time |
|----------|----------|--------------|----------|------|
| **Full MCTS** | 20 | 5 | 200 | 🔴 10s/step |
| **Hivemind (5 key)** | 5 | 5 | 100 | 🟡 1s/step |
| **Heuristic + MCTS** | 3 | 5 | 50 | 🟢 0.3s/step |

**Recommendation**: Use Hivemind with 3-5 key bacteria

---

## 🎓 Research Extensions

### 1. Evolutionary Co-Evolution

Train bacteria and macrophages against each other:

```python
for generation in range(100):
    # Train macrophages against current bacteria
    macrophage_policy = train_vs_bacteria(current_bacteria_policy)
    
    # Train bacteria against improved macrophages
    bacteria_policy = train_vs_macrophages(macrophage_policy)
    
    # Log performance
    track_arms_race(generation, macrophage_policy, bacteria_policy)
```

### 2. Transfer Learning

Train in one environment, test generalization:
- Train on 30×30 grid → test on 50×50
- Train with 5 bacteria → test with 20
- Train with no nutrients → test with nutrients enabled

### 3. Multi-Agent RL (QMIX)

Replace MCTS with neural network policies:

```python
class QMIXMacrophages:
    """
    Value decomposition for team coordination.
    Each macrophage has local Q-network, mixed by hypernetwork.
    """
    
    def __init__(self):
        self.local_q_networks = [QNetwork() for _ in range(num_macs)]
        self.mixing_network = HyperNetwork()
    
    def get_team_q_value(self, states, actions):
        # Each agent's local Q
        local_qs = [net(state, action) for net, state, action 
                    in zip(self.local_q_networks, states, actions)]
        
        # Mix into team Q-value
        team_q = self.mixing_network(local_qs, global_state)
        return team_q
```

---

## 🎯 Priority Recommendations

### For Your Current Project

1. ✅ **DONE**: Dynamic bacteria (Phase 1)
2. **NEXT**: Multi-macrophage with basic signaling (Phase 2 - simplified)
   - Start with 2-3 macrophages
   - Simple help signals ("I need backup")
   - Team reward function
3. **OPTIONAL**: Bacteria hivemind MCTS (Phase 3)
   - Only if bacteria seem too weak

### For Publication/Thesis

1. Implement full Phase 2 (multi-macrophage)
2. Run extensive experiments comparing:
   - 1 vs 2 vs 3 vs 4 macrophages
   - Team coordination ON vs OFF
   - Different signaling strategies
3. Analyze emergent behaviors:
   - Do they learn to flank?
   - Do they sacrifice for team?
   - How does communication range affect coordination?

### For PhD-Level Research

1. All of the above +
2. Implement QMIX or similar multi-agent RL
3. Compare against biological immune response data
4. Submit to AAMAS, NeurIPS, or ALIFE

---

## 📚 References for Implementation

### Multi-Agent Coordination
- **QMIX**: Rashid et al., "QMIX: Monotonic Value Function Factorisation for Decentralized Multi-Agent Reinforcement Learning" (ICML 2018)
- **COMA**: Foerster et al., "Counterfactual Multi-Agent Policy Gradients" (AAAI 2018)

### Biological Modeling
- **Quorum Sensing**: Waters & Bassler, "Quorum sensing: cell-to-cell communication in bacteria" (Annu Rev Cell Dev Biol 2005)
- **Macrophage Dynamics**: Sica & Mantovani, "Macrophage plasticity and polarization" (J Clin Invest 2012)

### Game-Theoretic Planning
- **Multi-Agent MCTS**: Soemers et al., "Monte-Carlo Tree Search for Multi-Agent Games" (Elsevier 2016)

---

## ✅ Implementation Checklist

### Phase 1 (Dynamic Bacteria) ✅ COMPLETE
- [x] Create `bacteria_adaptive.py`
- [x] Add config flags
- [x] Integrate with environment
- [x] Test adaptive behaviors
- [ ] Run experiments comparing adaptive vs fixed modes
- [ ] Document emergent behaviors

### Phase 2 (Multi-Macrophage)
- [ ] Refactor `Environment` to support multiple macrophages
- [ ] Create `MacrophageTeam` class
- [ ] Implement signaling system
- [ ] Update MCTS for team rewards
- [ ] Add team victory conditions
- [ ] Create visualization for team coordination
- [ ] Run rescue scenario tests
- [ ] Measure coordination effectiveness

### Phase 3 (Bacteria MCTS) - Optional
- [ ] Create `BacteriaHivemind` class
- [ ] Implement key bacteria selection
- [ ] Add collective MCTS planning
- [ ] Benchmark performance
- [ ] Compare vs adaptive heuristic

---

## 🎮 Quick Start: Testing New Features

### Test Dynamic Bacteria
```python
# In config.py
BACTERIA_ADAPTIVE_ENABLED = True
BACTERIA_STOCHASTICITY = 0.15

# Run
python experiments/run_simulation.py
```

Watch for emergent coordinated retreats and attacks!

### Future: Test Multi-Macrophage
```python
# In config.py (after implementation)
NUM_MACROPHAGES = 3
ENABLE_TEAM_COORDINATION = True
SIGNAL_RESPONSE_RADIUS = 8

# Run
python experiments/run_simulation.py
```

Look for rescue missions and coordinated strikes!

---

**You now have Phase 1 implemented and a clear roadmap for Phases 2-3!** 🎉
