# EHMbrAIn — speaker script

Companion to `EHMbrAIn-slides.pdf` (44 pages, 31 numbered content slides).
Slide numbers below are **PDF page numbers**.

**Target: 40 minutes + 10 for questions.** Section timings are per block.
Passages in _[brackets]_ are stage directions, not things to say.

**If you are short on time**, cut in this order: slide 20 (detectability),
slide 28 (sim-to-real), slide 33 (prognostic floor), slide 35 (economics).
Never cut 10, 11, 25 or 31 — the wall is the spine of the talk.

---

## Opening — slides 1–2 (2 min)

**Slide 1 (title).**

> This is a study about a comparison that the literature keeps getting wrong.
> Machine learning is now routinely applied to engine health monitoring, and it
> is routinely reported to beat the classical methods. What I will argue is that
> almost none of those comparisons are fair, that when you make one fair the
> answer splits in an interesting way, and that the split tells you exactly where
> to spend money.

**Slide 2 (roadmap).**

> I will work at two levels throughout. Every slide opens with the claim in one
> line, then shows the mechanism behind it. If you only track the one-liners you
> will get the argument; the rest is there for the people who want to check it.

_[Do not read the outline aloud. Point at it, one sentence, move on.]_

---

## Part 1 — The problem — slides 3–6 (4 min)

**Slide 4 (an engine degrades where nobody can look).**

> Here is the physical situation. A commercial turbofan stays bolted to a wing
> for years. In that time dust and pollution foul the compressor blades, the hot
> section creeps and oxidizes, tip clearances open, seals wear. None of it is
> visible. Nobody opens the engine to look.
>
> What the airline gets instead is this: a handful of numbers per flight. Shaft
> speeds, exhaust gas temperature, fuel flow. That is the whole window.
>
> _[point at the figure]_ And this is the one aggregate an operator actually
> watches — exhaust gas temperature margin, the distance to the certified limit.
> The sawteeth are compressor washes. When it hits zero the engine comes off the
> wing. A shop visit is millions of dollars; an unscheduled removal wrecks a
> fleet schedule for days.

**Slide 5 (two schools).**

> Two families of methods interpret that window.
>
> The classical one dates to the 1970s: gas path analysis. You build a
> thermodynamic model of the engine, and you invert it — from the symptoms, work
> backwards to which component degraded and by how much. Around that core the
> industry built a mature toolbox: baselines, trend smoothing, Kalman filters,
> expert rules.
>
> The second is machine learning, mostly in the last decade, mostly on remaining
> useful life, mostly on synthetic benchmarks.
>
> _[slow down here — this is the motivating claim of the whole talk]_ And here is
> the problem. AI is almost never compared against a traditional pipeline
> implemented with equal care — same data, same metrics, same tuning effort.
> What the field has is a stream of comparisons against straw men. Which means
> we do not actually know what AI contributes, under what conditions, or at what
> cost.

**Slide 6 (what this project builds).**

> So this project builds the fair comparison. Three pieces.
>
> A thermodynamic twin of a specific engine — the CFM56-7B, the one on most
> 737NGs — built only from certified public data. A hundred virtual engines run
> to failure, in the exact data format an airline receives. And both method
> families on identical data, identical metrics, identical splits, identical
> tuning budget, adjudicated under a protocol frozen before the results were
> seen.
>
> _[the key idea of the talk — land it]_ The reason to do this synthetically is
> on the bottom of the slide. On a real fleet you can ask "did the method predict
> well?". You cannot ask "did it blame the right component?", because nobody
> knows the true internal state of an engine on a wing. With synthetic ground
> truth you can. And attribution is exactly where classical GPA is weakest and
> where AI is least transparent — so it is the question worth building a whole
> testbed to answer.

---

## Part 2 — The physics — slides 7–11 (6 min)

_[This is the low-level foundation. If the audience is aerospace, go fast. If it
is a machine-learning audience, this is the part they need most.]_

**Slide 8 (the machine).**

> Very briefly, the machine. Air comes in through the fan. Most of it bypasses
> the core and produces most of the thrust. The rest is compressed, burned,
> expanded through two turbines that drive the compressors.
>
> The important structural fact is that there are two mechanically independent
> shafts — spools. The low-pressure spool, fan and booster driven by the
> low-pressure turbine, turning at a speed called N1. The high-pressure spool,
> compressor and turbine, at N2.
>
> The numbers along the bottom are station numbers; they will appear on later
> slides. And the sentence at the bottom is the constraint the whole talk lives
> under: the airline receives four channels. N1, N2, exhaust temperature, fuel
> flow. Some fleets buy an optional package with pressures and temperatures
> inside the core. Most do not.

**Slide 9 (the model to invert).**

> Now the mathematics, and it is short.
>
> We describe the internal condition of the engine with two numbers per
> turbomachine. Efficiency — how much of the ideal work it actually achieves;
> dirt and wear reduce it. And flow capacity — how much air it swallows; fouling
> closes a compressor down, erosion opens a turbine up. Five turbomachines, so
> ten numbers. That vector is what maintenance wants to know, and it is not
> measurable.
>
> What is measurable are deviations of the measured channels from what a healthy
> engine would read at the same operating condition. For the small deviations of
> in-service wear the relationship is linear: measurement deviations equal an
> influence matrix times the health vector, plus sensor bias, plus noise.
>
> _[pause]_ And now look at the shape of that problem. Ten unknowns. Three
> usable measurements — N1 does not count, it is the commanded thrust setting,
> not a symptom. So the matrix has rank three. There are infinitely many health
> states consistent with any measurement. The classical answer regularizes:
> penalize implausibly large deviations and pick one solution. That works, and
> it costs you accuracy in a way we will measure precisely.

**Slide 10 (smearing).**

> Here is what that ill-posedness does in practice, in a problem small enough to
> check by hand.
>
> Two health parameters, two measurements. The truth is a one percent efficiency
> loss in the high-pressure compressor. Feed the exact measurements into the
> inversion and you recover it exactly.
>
> Now perturb one measurement by a tenth of a percent — one sensor noise standard
> deviation. _[point]_ The answer becomes: compressor healthy, turbine degraded
> by one percent. The diagnosis has flipped to the wrong component entirely.
>
> The picture on the right is why. Each fault produces its own pattern of meter
> movements — think of it as an arrow in measurement space. Diagnosis asks which
> arrow the observed reading points along. When two arrows are nearly parallel,
> a noise-sized nudge moves the reading from one to the other. This is called
> smearing, and nothing is wrong with the algebra. The information simply is not
> in the measurements.

**Slide 11 (this engine's geometry).**

> That was a toy. Here is the real engine.
>
> With the cockpit sensor set, the smallest angle between two fault signatures on
> this engine is **1.3 degrees** — between high-pressure compressor efficiency and
> high-pressure turbine efficiency. Seven pairs sit below fifteen degrees, which
> is roughly where sensor noise blurs two faults into one. Rank three for ten
> unknowns.
>
> Add the four station probes and the rank goes to seven and the worst angle
> multiplies by five.
>
> _[this is the load-bearing claim of the talk — say it deliberately]_ And we
> repeated this across a six-point grid spanning the flight envelope. The
> structure barely moves: between 1.2 and 1.4 degrees everywhere. So the
> blindness is not a property of one flight condition you could escape by
> measuring somewhere else. It is intrinsic to the sensor set. Hold on to that,
> because everything in the second half of this talk is a consequence of it.

---

## Part 3 — The testbed — slides 12–16 (5 min)

**Slide 13 (the twin).**

> The twin is built in pyCycle, NASA's open cycle-modelling library, by morphing
> a validated example toward this engine — at most three parameters per step,
> requiring convergence at every step. That sounds fussy; it is the difference
> between a model and three weeks of debugging a Newton solver.
>
> Calibration is the part worth your attention. It uses only measured certified
> data: the type certificate, and the ICAO emissions databank, which publishes
> something quietly valuable — sea-level test-bed **measured fuel flows** at four
> thrust settings, for this exact engine variant, free.
>
> And we solve the anchor points thrust-matched. So the model's fuel flow is a
> **prediction**, confronted with a measurement it was never fitted to. That is
> the honest test, and it lands within one and a half to three percent at takeoff,
> climb-out and approach. Cruise fuel consumption comes out within two percent of the
> published value without ever being a target.

**Slide 14 (the fleet).**

> From that twin we generate a hundred engines living full lives. Six degradation
> mechanisms, each a closed-form function of flight cycle, with magnitudes from
> the gas-turbine degradation literature.
>
> _[point across the panels]_ Fouling saturating toward an asymptote with wash
> recoveries. Linear erosion. Tip clearance opening fast then slow. Hot section
> accelerating late — efficiency down, flow capacity up, that pair is the
> signature of an eroded turbine nozzle. Foreign object damage as Poisson steps.
> And discrete acute fault episodes.
>
> The detail that matters later: each mechanism's contribution is stored
> separately. So the fleet records not just how degraded an engine is, but by
> what.

**Slide 15 (making it hard).**

> Now, a synthetic dataset's most common disease is being accidentally easy, and
> a benchmark that flatters your method is worthless. So we set a gate in
> advance: a deliberately weak classifier — plain logistic regression on raw
> channels — must **fail** to solve the isolation task, below sixty percent.
>
> First pass: it scored just under eighty-two percent. The gate failed.
>
> _[this is a credibility moment — do not rush it]_ We diagnosed why. Our fault
> labels were the dominant chronic mechanism, and chronic mechanisms co-evolve
> with age in every engine. So the label was an age proxy, and raw channel levels
> encode age directly. The task we had defined was not fault isolation at all.
>
> There were two available responses: relax the threshold, or fix the design. We
> redesigned the dataset — discrete single-parameter fault episodes drawn from
> the confusable pairs — and it passed at fifty-four percent. Then on a later
> fleet version the gate bit **again** at sixty-two percent, and we scaled the
> fault magnitudes down until it passed.
>
> Both failures are in the report in full, because a gate that cannot fail
> certifies nothing.

**Slide 16 (safeguards).**

> Four safeguards, all fixed before any method ran.
>
> Pre-registration: hypotheses, thresholds, splits with file hashes, statistical
> tests, all frozen under a git tag before the confirmatory run. Borrowed from
> clinical trials — you state what would convince you before you look.
>
> Symmetric tuning: fifty automated search trials each. This one matters more
> than it sounds. Any asymmetry in effort silently converts a method comparison
> into an effort comparison, and that is precisely the literature's disease.
>
> Split by engine, never by flight — flights of one engine are near-duplicates,
> and splitting by flight lets a model memorize individual engines.
>
> And a genuinely strong baseline: the traditional pipeline built to industrial
> shape, every expert rule justified.
>
> The block at the bottom is the discipline that makes the rest trustworthy.
> Numbers exist in two castes and never mix. Exploratory numbers, produced while
> building. Confirmatory numbers, produced once, after the freeze, on data
> touched exactly once. Only the second kind supports a verdict.

---

## Part 4 — What classical GPA delivers — slides 17–21 (5 min)

_[This section is new relative to how this work is usually presented, and it is
the quantitative floor for everything after. Do not skip 19.]_

**Slide 18 (the experiment).**

> Before comparing anything to anything, a question nobody in this literature
> seems to answer: what does the textbook method actually deliver on this engine?
>
> The design matters. For the forward direction — playing the engine — we use a
> **nonlinear** neural twin of the thermodynamic model, with the full curvature of
> the physics. For the inverse — playing the monitoring system — we use the
> **linear** regularized least squares on the influence matrix. Two different
> instruments on purpose: inverting the same matrix that generated the data would
> be circular and exact by construction, and would tell us nothing.
>
> Here is one run. The compressor loses one percent of efficiency. The cockpit
> sees fuel flow up half a percent, temperature up half a percent, core speed
> down a tenth. Invert that, and you get back **a quarter** of the fault. Repeat
> with fresh sensor noise five hundred times and the method names the right
> component **twenty-two percent** of the time. Its favourite wrong answer is the
> turbine.

**Slide 19 (what the inversion gives back).**

> Systematically now, one fault at a time, both sensor sets.
>
> Left panel, the magnitude that comes back. With the cockpit set, between two
> and seventy-six percent depending on the parameter. With station probes, thirty
> to ninety.
>
> Right panel is the one to look at. The smearing index — the share of the
> diagnosed magnitude that lands on components that are in fact healthy. Median
> **0.85**. Six sevenths of what the method reports is attributed to the wrong
> place. And notice the station probes barely improve it: more sensors buy you
> magnitude, not tidiness, because the estimator still spreads what it cannot
> separate.
>
> The extreme case is worth naming. Turbine flow capacity: the cockpit recovers
> **one point nine percent** of it — the fault is effectively invisible — and the
> extended set recovers ninety. That is a forty-seven-fold difference bought by
> one pressure probe at the compressor exit.

**Slide 20 (how small a fault can be named).** _[cuttable]_

> Turning that into the number an operator would ask for: how small a fault can
> this sensor package still name correctly, nine times out of ten?
>
> Cockpit set: it manages that for **one** of the ten parameters. The extended
> set manages seven, several from below one percent.
>
> And the faults in our fleet are between 0.3 and 1.1 percent. So most faults in
> this benchmark sit below the cockpit set's resolution limit — which is going to
> explain, in about five minutes, why the isolation contest ends the way it does.

**Slide 21 (a real engine).**

> Single faults are the textbook exercise. Real engines deteriorate everywhere at
> once, so here is a real fleet engine late in life, diagnosed.
>
> Black is truth. The engine is dominated by hot-section damage: turbine
> efficiency down four percent, and the flow capacity opened by two point four.
>
> The cockpit diagnosis, in red, gets the headline right — turbine efficiency
> down three point three. And then it fails on the physics that would confirm it:
> the flow opening comes back as **zero point six** instead of two point four. So
> the classic eroded-nozzle fingerprint is simply not visible in the diagnosis.
> The extended set, in blue, recovers it.
>
> _[the honest part — say it]_ And both of them invent damage that is not there.
> True fan efficiency loss is under half a percent; both report three times that.
> True low-pressure turbine flow is essentially zero; both report about minus
> one point three — wrong in sign as well as size. A maintenance decision taken
> on those two numbers opens a healthy module.

---

## Part 5 — The contest — slides 22–28 (7 min)

**Slide 23 (five verdicts).**

> Now the comparison itself. Both families tuned under the frozen fifty-trial
> budget, the twenty test engines evaluated exactly once, Wilcoxon paired by
> engine, exact McNemar for the binary outcomes, Holm correction across
> hypotheses.
>
> Three confirmed, two refuted. _[let them read for a beat]_ And I want to be
> clear that the two refutations are not failures of the study — they are two of
> its four most useful results. In fact the two refutations did more work than
> the three confirmations: the confirmations said roughly what the field
> expected — learning predicts life better, detects earlier, quantifies
> uncertainty better. It is the refutations that changed our recommendations. I
> will take them in order.

**Slide 24 (H1 — detection).**

> Detection first, and the story here is about method, not about models.
>
> Before tuning, our AI detector scored recall 0.06 and we wrote in the report
> that this looked like a design gap rather than a threshold gap. The frozen
> search space happened to contain the suspected lever — the length of the
> feature windows. Given fifty trials the tuner stretched those windows, and
> recall went from 0.06 to **0.48**, at a median delay of 499 cycles. Twelve
> times earlier than the tuned classical detector, at the same false alarm count.
>
> Two things transfer out of this. Window length is the first knob any change
> detector lives or dies by. And — the uncomfortable one — the traditional side's
> tuned configuration was selected on a validation split with only two clean
> engines, and it generalized worse than its own untuned defaults. Threshold
> selection on small clean samples is fragile for **any** method. Budget clean
> engines accordingly.

**Slide 25 (H2 refuted).**

> Isolation. This is the hypothesis we most expected to confirm, and it is
> refuted.
>
> On the thirteen confusable test episodes the two families tie exactly.
> Thirty-one percent each. And the disagreement pattern is perfectly symmetric —
> four episodes only the AI gets right, four only the classical rule gets right.
> It is very close to a coin flip about which one lands the point.
>
> _[this is the central conclusion of the talk]_ Now, we predicted this before
> any model ran, from the influence matrix geometry: signatures 1.3 degrees apart
> in a rank-three measurement space. Both families are bounded by the same
> information, and no amount of modelling recovers information the sensors never
> captured.
>
> Which turns a software question into a purchasing question. If confusable fault
> isolation matters to your operation, the fix is instrumentation.

**Slide 26 (H3 and H5 — prognosis).**

> Prognosis is where learning earns its keep, decisively.
>
> Remaining-life error of 858 cycles at ninety percent of life against 1981 for
> the tuned classical extrapolation. Effect size 0.89 — pick any engine's error
> from each method at random and the classical one is larger about ninety-four
> percent of the time.
>
> _[state both scales — do not quote only the big one]_ And be honest about which
> baseline that is against. The 2.3-fold figure is against the linear
> extrapolation an operator fields today. Against the best advanced classical
> prognostic — similarity-based, which can follow the curve — it narrows to about
> 1.3 times. Both are true. Quoting only the first would be exactly the straw man
> this whole project criticizes.
>
> But look at the shape rather than the number. _[point]_ The classical errors
> are wide and skewed **optimistic** — it predicts more life remaining than there
> is, consistently, which is the dangerous direction. The AI's are tight and
> near-unbiased. The mechanism is simple: a straight line cannot bend to follow
> accelerating hot-section decay, and a sequence model learns that curvature from
> data.
>
> And the uncertainty statement that comes with it — this is hypothesis five. The
> conformal interval covers eighty-eight percent empirically against a ninety
> percent nominal, at plus-minus about two thousand cycles. The classical band
> over-covers at ninety-eight percent but is plus-minus eleven and a half
> thousand. That is the difference between a plannable shop visit slot and a
> shrug.

**Slide 27 (H4 refuted).**

> The second refutation, and the one I would most want a practitioner to
> remember.
>
> "Physics-informed" is a popular label. The obvious implementation is to feed
> your physics-based state estimates into the network alongside the raw data. We
> pre-registered the prediction that this would help most when data is scarce.
>
> It made things **worse at every data fraction**, and most worse exactly where we
> predicted it would help most. The mechanism is intelligible in hindsight: those
> state estimates are heavily smeared. They are ten noisy, mutually correlated
> channels whose information content is already present in the three deviations
> they were computed from. With seven training engines, ten extra input
> dimensions are pure variance.
>
> We later re-opened this a second, independent way — different feature, different
> fleet, nonlinear physics — and refuted it again. Two negatives. Physics
> injection needs an information channel the raw data lacks; concatenating
> estimator outputs is not one.

**Slide 28 (sim-to-real).** _[cuttable]_

> The obvious objection to everything so far is that we built the world these
> methods are being scored in. So we re-ran the prognosis question on NASA's
> C-MAPSS, the field's reference benchmark, which we had no hand in generating.
>
> Same direction, comparable magnitude: 14.9 cycles against 42.1. The ranking is
> not an artifact of our simulator.
>
> One disclosure on the right: our plan named the larger successor benchmark,
> whose multi-gigabyte distribution breaks this project's rule that everything
> must run on a desktop. We substituted, and we say so in the decision register
> rather than quietly changing the plan.

---

## Part 6 — Beyond the wall — slides 29–35 (7 min)

**Slide 30 (observability without hardware).**

> The isolation result ended with "buy sensors", which is an expensive
> conclusion. So we looked for cheaper exits, and found two.
>
> The influence matrix is not one matrix — it changes with flight condition. So a
> window of ordinary snapshots at scattered conditions is a set of different
> projections of the same health state, and fusing them is a tomography problem.
>
> The audit was two-sided. _[point at figure]_ Free scatter — the ambient and
> power variation you get for nothing — restores algebraic rank immediately, but
> the angles saturate almost at once. Free variation is real but thin.
>
> So: exit one. Treat the **schedule of routine reports** as an observability
> design variable, which as far as we know is a new framing. Adding one periodic
> stabilized low-power cruise report buys seventy-nine percent more separability
> on the turbine ambiguity. The cost is a procedure bulletin, not avionics.
>
> Exit two. Fuse the window with a learner, but keep the physics inversion. A
> network fed per-flight physics inversions isolates at 0.65 against classical
> stacking's 0.17. The same network fed raw sequences fails outright at 0.12.
> The division of labour is the lesson: let physics do the inversion it does
> well, let the network do the temporal fusion it does well.
>
> And what no schedule fixes: the compressor-versus-turbine efficiency pair stays
> below 1.8 degrees everywhere in the envelope. That one is fundamental.

**Slide 31 (only a real sensor).**

> Which raises the question a vendor will ask you: can we not just **predict** the
> missing sensor from the ones you have? We have a calibrated twin; let it
> compute a virtual pressure probe.
>
> Three conditions, pre-registered. Cockpit only: 0.15. Cockpit plus **virtual**
> station channels: 0.15, p equals one. Cockpit plus **real** station channels:
> **0.92**.
>
> This is the data-processing inequality made operational. A quantity computed
> from the cockpit data cannot carry information the cockpit data did not already
> hold, so it cannot separate faults the cockpit cannot separate. No software
> vendor, however sophisticated, substitutes for the missing measurement.
>
> The practical note is at the bottom, and it is unusually good news: the
> parameters that break the wall are already measured by the engine's control
> computer. The barrier is recording and downlinking them, not installing
> hardware.

**Slide 32 (the certificate).**

> This is the part I would most like to survive the talk.
>
> Every gas path diagnosis in industry is a point estimate that hides what it
> cannot know. So: accumulate Fisher information over the conditions an
> individual engine actually flew, through the calibrated twin. Invert it. You
> get the Cramér-Rao bound — the best per-direction precision **any** unbiased
> estimator can achieve from that engine's data. That is a per-engine,
> history-resolved certificate of what its diagnosis can honestly claim.
>
> Observability analyses of gas turbines exist. What has never been possible is
> the right-hand panel: **checking that the certificate is honest**, because on a
> real engine the true component state is unknowable.
>
> Here it is knowable. The certified precision ranks the filter's actual error
> against ground truth with a correlation of 0.70. The directions the certificate
> calls identifiable are the ones the estimator gets right; the ones it calls
> unobservable are where the estimate is worthless. To our knowledge that is the
> first ground-truth-validated identifiability guarantee for gas path diagnosis —
> and it is a validation only a synthetic-truth benchmark can perform.

**Slide 33 (the prognostic floor).** _[cuttable]_

> The same trick, applied to the future instead of the present.
>
> Before an operator funds a better prognostic, they should ask: is my error large
> because my method is weak, or because the future is genuinely unknowable from
> what I can see today? Those have opposite remedies, and confusing them either
> wastes budget or abandons a target that was in reach.
>
> So we condition on each engine's **true** current health, find its nearest
> neighbours in health space — engines in a near-identical present condition — and
> measure the spread of their true remaining lives. Two engines in the same state
> that fail hundreds of cycles apart differ for reasons no present measurement
> contains. That spread is an irreducible floor.
>
> At mid-life, **eighty-seven percent** of the remaining-life uncertainty is
> irreducible. Remaining life is, to first order, not written in the present —
> which is a sober correction to a lot of prognostics work. But late in life a
> four-fold reducible gap opens. That is where a better prognostic pays, and only
> there.

**Slide 34 (the determinability map).**

> One more axis, because everything so far lives at a single sensor-quality
> setting. We regenerated the fleet with sensor noise at half, one, two and four
> times nominal — same seed, same engines, same faults, only the measurement
> quality moves — and re-scored both tuned families.
>
> Three findings, left to right.
>
> Prognosis is essentially noise-proof: the AI's error is flat across a factor of
> eight in noise while the classical extrapolation degrades by forty percent. The
> advantage **widens** from 2.3 to 3.4 times. So the business case does not depend
> on sensor quality.
>
> Middle panel is the one that closes the argument of this talk. Halving every
> sensor sigma — an expensive upgrade — leaves confusable isolation at 0.31 and
> 0.38. Nowhere near the 0.92 that **different** sensors deliver. Two signatures
> 1.3 degrees apart stay 1.3 degrees apart however precisely you measure them.
> So the purchase order is specific: probes at new stations, not better probes at
> the old ones.
>
> And on the right, the classical detector falls off a cliff — recall 0.48 down to
> zero — while the AI degrades gracefully. Robustness to instrumentation quality
> may be learning's most transferable advantage here.

**Slide 35 (economics).** _[cuttable]_

> Briefly, what it is worth, and this chapter leads with its caveats on purpose.
>
> We monetize exactly **one** mechanism — the one the results support: less biased
> prognosis converting unscheduled removals into scheduled ones. Everything else
> an enthusiast might count is discussed and deliberately left out of the total.
>
> For a hundred-engine operator, median three million dollars a year. But read
> the spread, not the median: tenth percentile 0.9 million, ninetieth 8 million,
> and net-negative in one percent of draws.
>
> And the right-hand column is why it could be nothing: the isolation wall is
> unbroken, so an operator hoping for better diagnosis from this investment would
> be disappointed; false alarms erode trust and money; and the sim-to-real check
> preserved the ranking but not necessarily the magnitude.

---

## Part 7 — Takeaways — slides 36–40 (4 min)

**Slide 37 (the split verdict).**

> So, the answer to the question the talk started with.
>
> Learning wins where the task is statistical aggregation over time. Prognosis —
> 2.3 times better than fielded practice, 1.3 times better than the best classical
> method, and robust across three architectures and across sensor quality.
> Detection timing. Calibrated uncertainty.
>
> Nobody wins where the task is bounded by what the sensors observed. Confusable
> isolation is a tie, and the fix is seventy-seven percentage points from real
> station probes, and exactly zero from virtual ones.
>
> _[the reframe — this is the closing idea]_ And I want to be careful about what
> that says regarding physics, because it is easy to misread. Physics did not
> lose. Injecting physics into the learner failed, twice. But physics **diagnosed
> the limits**: the influence matrix geometry predicted, before any learning ran,
> exactly where every method would fail — and it was right. Using physics to know
> what is knowable succeeded completely.

**Slide 38 (Monday).**

> If you take one slide away, take this one. In order of how much money each
> moves.
>
> Deploy learning for prognosis first — that is the business case. Detection:
> modern change detection with tuned windows; the win came from window search,
> which is cheap and auditable, not from depth. Isolation: buy sensors, not
> models — and specifically probes at new stations. Redesign your report schedule
> before buying anything. Protect the sensor estate; a drifting thermocouple
> corrupts every consumer downstream, learned or classical. Distrust untuned
> comparisons in either direction. And demand pre-registration of vendor claims.
>
> _[the last one, with a beat before it]_ A split verdict is what honest
> evaluation looks like. If a vendor shows you a clean sweep, that should raise
> your suspicion, not your confidence.

**Slide 39 (what transfers).**

> Finally, the intellectual honesty this kind of work requires.
>
> What transfers off a simulator is rankings, mechanisms and geometry — which
> method wins where and why, and the observability structure any engine of this
> class shares. Every operational claim I have made is of that kind.
>
> What does **not** transfer is absolute numbers. An error of 858 cycles is a
> property of this simulated world and its noise settings. I am not claiming your
> fleet will see that.
>
> And the whole evidence base regenerates in about eleven minutes on one desktop
> machine. Every figure in these slides came out of a script; none was copied by
> hand.

**Slide 40 (standout).**

> _[pause. Let the slide sit for two seconds before speaking.]_
>
> The split verdict is the result. A method comparison that produces a clean
> sweep is usually measuring effort, not method.
>
> Thank you — happy to take questions.

---

## Question preparation

The likely challenges, and honest answers. Do not bluff; every one of these has
a real answer in the work.

**"Your traditional baseline is a straight line. That is a straw man."**
> Fair, and we tested it. The linear extrapolation is what airlines actually
> field, so it is the right baseline for an industry-transfer claim — but it is
> not the research frontier. So we implemented a similarity-based prognostic, the
> C-MAPSS-standard classical method, which can follow nonlinear decay. It nearly
> halves the classical error: 1981 down to 1118. The AI still wins at 858. So the
> honest statement is two-scaled: 2.3 times better than what operators field
> today, 1.3 times better than the best classical method. A thirty percent edge,
> not an order of magnitude.

**"It is all synthetic. Why should I believe any of it?"**
> Four defences, all built before the headline experiments. The generator is
> anchored to certified measurements it was never fitted to. The dataset had to
> pass a difficulty gate that it failed twice and forced redesigns. The central
> conclusion was re-tested on a benchmark we did not generate. And we state
> explicitly that absolute numbers do not transfer — only rankings, mechanisms
> and geometry do.

**"Twenty test engines. Thirteen confusable episodes. That is nothing."**
> Correct, and it is our main statistical limitation. It is why every test is
> paired by engine, why we use exact rather than asymptotic tests, and why we
> report effect sizes and bootstrap intervals rather than p-values alone. It is
> also why one of our own noise-axis hypotheses was refuted on a threshold that
> was finer than the sample could resolve — we report that rather than hide it.

**"You refuted your own hypotheses. Is that not a failed project?"**
> The opposite. Pre-registration is what makes a refutation credible, and the
> refutations are the results that changed our recommendations. H2 turned a
> software question into a purchasing question. H4 killed a popular technique on
> this task, twice. A study that confirms everything it predicted is usually a
> study that moved its goalposts.

**"Why not a large modern architecture?"**
> We tested that the result is not an artifact of the architecture: a temporal
> convolutional network and a Transformer both beat the classical baseline by
> essentially the same margin as the recurrent model. Architecture choice barely
> matters here, and bigger networks did not help — the ceiling is the information
> in the measurements, not the size of the model.

**"Would this work on an engine you have no data for?"**
> The recipe would. The twin is built only from a type certificate and a public
> emissions databank, both of which exist for every certified engine. That is
> deliberate: the method is portable even where our specific numbers are not.

**"What is the single most novel thing here?"**
> The certificate, on slide 32. Observability analyses exist; a per-engine,
> history-resolved identifiability guarantee that has been **proven honest against
> the true component state** does not, because on real hardware that check is
> impossible. That is the one contribution that needed this whole testbed to
> exist.
