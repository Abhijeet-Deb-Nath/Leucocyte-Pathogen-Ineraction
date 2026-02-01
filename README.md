# Immune System vs Bacteria Multi-Agent Simulation

## 🎯 Project Overview

This project simulates the adversarial interaction between **macrophages** (immune cells) and **Pseudomonas aeruginosa** bacteria using advanced AI decision-making algorithms. The simulation models realistic biological behaviors including bacterial quorum sensing, biofilm formation, virulence factors, and macrophage exhaustion.

### Core Concept
- **Macrophages**: Intelligent agents using Monte Carlo Tree Search (MCTS) to hunt and eliminate bacteria
- **Bacteria**: Adaptive adversarial agents with collective behaviors (clustering, biofilm, virulence)
- **Environment**: 30x30 grid with nutrients, spatial dynamics, and health systems
- **Objective**: Compete until one side achieves victory (elimination, survival, or timeout)

---

## 🧬 Biological Features Modeled

### Bacterial Intelligence (Pseudomonas aeruginosa)
- **Quorum Sensing**: Bacteria communicate when clustered, forming defensive biofilms
- **Biofilm Formation**: Reduces damage taken and increases attack power
- **Virulence Factors**: Mature bacteria produce toxins that passively damage nearby macrophages
- **Dynamic Replication**: Nutrient-boosted bacteria replicate faster
- **Age & Maturity**: Young bacteria are weak; mature bacteria attack effectively

### Macrophage Capabilities
- **Vision System**: Detects bacteria within radius to plan strategic moves
- **Toxin Burst**: Area-of-effect attack with cooldown mechanic
- **Exhaustion System**: Fighting multiple bacteria causes fatigue and vulnerability
- **MCTS Planning**: Simulates future scenarios to choose optimal actions

### Environmental Dynamics
- **Nutrient Patches**: Bacteria consume nutrients to heal and boost replication
- **Spatial Strategy**: Movement, positioning, and clustering affect outcomes
- **Time Pressure**: Limited time horizon creates urgency

---

## 🚀 Current Capabilities

### Implemented Features
✅ **Single Macrophage vs Bacterial Colony**  
✅ **MCTS-based Macrophage AI** with rollout simulations  
✅ **4 Bacterial Behavior Modes**: `scatter`, `cluster`, `defend`, `replicate`  
✅ **Quorum Sensing & Biofilm** mechanics  
✅ **Virulence Factor** toxin production  
✅ **Macrophage Exhaustion** from prolonged combat  
✅ **Real-time PyQt5 Visualization**  
✅ **Data Logging & Analysis** (CSV export, heatmaps, performance plots)  
✅ **Parameter Sweeping** for experimentation  

### Behavior Modes (Current)
Bacteria currently use **static modes** defined in `config.py`:
- **`scatter`**: Random exploration
- **`cluster`**: Group around center of mass
- **`defend`**: Aggressive pursuit when macrophage nearby, otherwise cluster
- **`replicate`**: Prioritize nutrient consumption

---

## 🔮 Roadmap: Advanced Features

### Phase 1: Dynamic Bacteria Intelligence ⏳
**Goal**: Replace hardcoded behavior modes with adaptive, condition-based decision-making

**Implementation**:
```python
# Bacteria choose behavior based on context
def choose_bacteria_strategy(bacteria, env):
    if bacteria.health < 10 and nearby_nutrients:
        return "seek_nutrient"
    elif macrophage_distance < 5 and in_biofilm:
        return "attack"
    elif macrophage_distance < 3 and not in_biofilm:
        return "flee"
    elif quorum_threshold_met:
        return "cluster_biofilm"
    else:
        return "explore" + stochastic_noise()
```

**Benefits**:
- More realistic emergent behavior
- Bacteria adapt to changing threats
- Retains stochasticity (not fully deterministic)

**Complexity**: ⭐⭐ (Low - straightforward refactor)

---

### Phase 2: MCTS for Bacteria 🔬
**Goal**: Give bacteria the same planning capabilities as macrophages

**Challenges**:
1. **State Space Explosion**: 20+ bacteria × 5+ actions = huge search tree
2. **Coordination Problem**: Who decides for the collective?
3. **Computational Cost**: Rollouts multiply by number of bacteria

**Solutions**:
- **Option A**: Single "Bacteria Hive Mind" agent controls all bacteria (tractable)
- **Option B**: Each bacterium runs lightweight MCTS (expensive but decentralized)
- **Option C**: Representative sampling (3-5 key bacteria plan, others follow)

**Recommendation**: Start with **Option A** (collective MCTS) for feasibility

**Complexity**: ⭐⭐⭐⭐ (High - significant architectural change)

---

### Phase 3: Multi-Macrophage Cooperation 🤝
**Goal**: Multiple macrophages signal for help and coordinate attacks

**New Mechanics**:
- **Signaling System**: Macrophages broadcast "need help" when overwhelmed
- **Proximity Response**: Nearby macrophages adjust priorities
- **Coordinated Attack**: Pincer movements, flanking strategies
- **Global Objective**: Maximize **team survival**, not individual kills

**Example Scenario**:
```
Macrophage A: "Bacteria cluster at (10,15), health critical!"
Macrophage B: Calculates: "I can reach in 3 steps, worth diverting"
Macrophage C: "Too far, continuing patrol"
→ A+B coordinate to eliminate cluster before A dies
```

**Implementation Changes**:
1. **Environment**: Track multiple macrophages
2. **MCTS Reward**: Team-based scoring (sum of team health - bacteria count)
3. **Communication Protocol**: Shared information during planning
4. **Action Space**: Add "wait for backup" action

**Complexity**: ⭐⭐⭐⭐⭐ (Very High - major redesign)

---

### Phase 4: Team vs Team Global Optimization 🏆
**Goal**: Both sides optimize for **collective survival** using multi-agent algorithms

**Approaches**:

**For Macrophages**:
- **Centralized MCTS**: Single tree search for all macrophages (like hivemind)
- **Decentralized MCTS with Communication**: Each agent shares rollout information
- **Multi-Agent RL**: Train neural networks with shared reward (QMIX, COMA)

**For Bacteria**:
- **Swarm Intelligence**: Emergent coordination without central planning
- **Collective MCTS**: Search over joint action spaces
- **Evolutionary Strategies**: Bacteria evolve coordinated tactics

**Key Concept**: 
Replace individual fitness with **team fitness**:
```python
# Old: Individual reward
reward = my_health - bacteria_near_me

# New: Team reward
reward = sum(all_macrophage_health) - total_bacteria_count
```

**Complexity**: ⭐⭐⭐⭐⭐⭐ (Research-level difficulty)

---

## 🛠️ Implementation Recommendations

### **What You Should Do First**

#### ✅ **Phase 1: Dynamic Bacteria Behavior** (DO THIS NEXT)
- **Time**: 2-4 hours
- **Value**: High (more realistic, publishable)
- **Risk**: Low (small refactor)

**Steps**:
1. Create `agents/bacteria_adaptive.py`
2. Implement context-based strategy selection
3. Add small random noise to decisions
4. Test with experiments

#### ⏸️ **Phase 2: Bacteria MCTS** (OPTIONAL - LATER)
- **Time**: 2-3 days
- **Value**: Medium (interesting but computationally expensive)
- **Risk**: Medium (might be too slow)

**Recommendation**: Only pursue if Phase 1 shows bacteria are too dumb

#### ✅ **Phase 3: Multi-Macrophage** (DO THIS FOR THESIS/PUBLICATION)
- **Time**: 1-2 weeks
- **Value**: Very High (novel research contribution)
- **Risk**: High (major architectural changes)

**Recommendation**: This is your **killer feature** - makes project unique

#### 🔬 **Phase 4: Team RL** (RESEARCH PROJECT)
- **Time**: Months
- **Value**: Extreme (PhD-level work)
- **Risk**: Very High

**Recommendation**: Only if you want to publish in top-tier conference

---

## 📁 Project Structure

```
AI/
├── config.py                    # All simulation parameters
├── simulation_history.csv       # Latest run data
├── agents/
│   ├── heuristic.py            # Greedy macrophage strategy
│   ├── mcts.py                 # Monte Carlo Tree Search agent
│   └── bacteria_adaptive.py    # [TODO] Dynamic bacteria AI
├── simulator/
│   ├── entities.py             # Macrophage & Bacteria dataclasses
│   └── environment.py          # Grid, rules, step logic
├── experiments/
│   ├── run_simulation.py       # Single run with GUI
│   └── param_sweep.py          # Batch experiments
└── visualization/
    ├── pyqt_gui.py             # Interactive real-time GUI
    ├── plots.py                # Performance charts
    └── animation.py            # Animated visualizations
```

---

## 🎮 Usage

### Run Single Simulation with GUI
```bash
cd experiments
python run_simulation.py
```
- **Controls**: Watch real-time battle unfold
- **Output**: CSV history, heatmaps, performance plots

### Run Parameter Sweep (Batch Testing)
```bash
cd experiments
python param_sweep.py
```
- Tests multiple configurations
- Generates comparative analysis

### Configure Simulation
Edit `config.py`:
```python
GRID_SIZE = 30                   # Battlefield size
ROLLOUT_BUDGET = 200             # MCTS simulations per decision
BACTERIA_MODE = "defend"         # Behavior: scatter/cluster/defend/replicate
QUORUM_SENSING_ENABLED = True    # Enable biofilm mechanics
```

---

## 🧪 Key Parameters (Tunable)

### Macrophage
- `MACROPHAGE_HEALTH` = 100
- `MACROPHAGE_VISION_RADIUS` = 6
- `MACROPHAGE_TOXIN_RADIUS` = 2 (damage radius)
- `MACROPHAGE_TOXIN_COOLDOWN` = 3 steps
- `ROLLOUT_BUDGET` = 200 (MCTS simulations)

### Bacteria
- `INITIAL_BACTERIA_COUNT` = 5
- `BACTERIA_ATTACK_PROB` = 0.3
- `BACTERIA_REPLICATION_PROB` = 0.02
- `QUORUM_THRESHOLD` = 3 (bacteria needed for biofilm)
- `BIOFILM_DEFENSE_BONUS` = 5 (damage reduction)

### Victory Conditions
- Macrophage wins: Eliminate all bacteria
- Bacteria win: Kill macrophage OR survive 300 steps OR reach 40 population

---

## 📊 Output & Analysis

### Generated Files
- `simulation_history.csv`: Step-by-step metrics
- `performance_*.png`: Health/population over time
- `heatmap_macrophage.png`: Movement patterns
- `heatmap_toxin.png`: Attack locations

### Metrics Tracked
- Macrophage health trajectory
- Bacteria population dynamics
- Nutrient consumption rates
- Spatial distributions

---

## 🔬 Research Potential

### Current State
- Single-agent MCTS vs scripted adversary
- Biological realism with game theory

### Future Directions
1. **Multi-Agent Coordination**: Cooperative immune response
2. **Adaptive Adversaries**: Bacteria learn counter-strategies
3. **Evolutionary Arms Race**: Co-evolution of both sides
4. **Transfer Learning**: Train agents in one environment, test in another
5. **Real-World Validation**: Compare to biological infection data

### Publication Opportunities
- **Conferences**: AAMAS, NeurIPS (multi-agent track), ALIFE
- **Journals**: Artificial Life, JAIR, IEEE TCIAIG
- **Themes**: Bio-inspired AI, adversarial planning, swarm intelligence

---

## 🤔 Feasibility Assessment

| Feature | Difficulty | Time | Worth It? |
|---------|-----------|------|-----------|
| Dynamic Bacteria | ⭐⭐ Low | 4 hours | ✅ Yes - Easy win |
| Bacteria MCTS | ⭐⭐⭐⭐ High | 3 days | ⚠️ Maybe - If needed |
| Multi-Macrophage | ⭐⭐⭐⭐⭐ Very High | 2 weeks | ✅ YES - Best feature |
| Team RL | ⭐⭐⭐⭐⭐⭐ Research | Months | 🔬 PhD-level |

---

## 🎯 Recommended Next Steps

### For a Strong Project (1-2 weeks)
1. ✅ **Implement Dynamic Bacteria** (Phase 1)
2. ✅ **Add Multi-Macrophage** (Phase 3 - simplified version)
3. ✅ **Run comprehensive experiments** comparing strategies
4. ✅ **Write analysis** of emergent behaviors

### For a Research Publication (2-3 months)
1. All of the above +
2. Implement team-based MCTS for macrophages
3. Add bacterial swarm intelligence
4. Compare to biological infection models
5. Submit to AAMAS or ALIFE

---

## 📚 Dependencies

```bash
pip install numpy pandas matplotlib seaborn PyQt5
```

---

## 👨‍💻 Author Notes

This simulation demonstrates:
- **Game-theoretic AI**: Adversarial planning with MCTS
- **Biological modeling**: Realistic microbial behaviors
- **Emergent complexity**: Simple rules → complex dynamics
- **Research platform**: Extensible for novel multi-agent algorithms

**The multi-macrophage cooperation feature is the most novel contribution** - few projects model immune cell coordination with game-theoretic planning.

---

## 📄 License

[Add your license here]

---

## 🙋 Questions?

This is a **living project** with room for expansion. The architecture supports adding:
- More agent types (neutrophils, dendritic cells)
- Environmental hazards (inflammation, tissue damage)
- 3D spatial modeling
- Real-time strategy game interface

**Start with Phase 1 (dynamic bacteria), then move to Phase 3 (multi-macrophage) for maximum impact!**
