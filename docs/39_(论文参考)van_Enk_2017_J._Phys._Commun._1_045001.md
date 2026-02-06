\section*{OPEN ACCESS}

\section*{RECEIVED}

14 June 2017

\section*{REVISED}

9 August 2017

\section*{ACCEPTED FOR PUBLICATION}

4 October 2017

\section*{PUBLISHED}

8 November 2017

Original content from this work may be used under the terms of the Creative Commons Attribution 3.0 licence.
Any further distribution of this work must maintain attribution to the author(s) and the title of the work, journal citation and DOI.

\section*{PAPER}

\title{
Photodetector figures of merit in terms of POVMs
}

\author{
SJ van Enk \\ Department of Physics and Oregon Center for Optical, Molecular \& Quantum Sciences University of Oregon, Eugene, OR 97403, United States of America \\ E-mail: svanenk@uoregon.edu
}

Keywords: photodetection, measurement, photons

\begin{abstract}
A photodetector may be characterized by various figures of merit such as response time, bandwidth, dark count rate, efficiency, wavelength resolution, and photon-number resolution. On the other hand, quantum theory says that any measurement device is fully described by its positive-operatorvalued measure (POVM) which generalizes the textbook notion of the eigenstates of the appropriate hermitian operator (the 'observable') as measurement outcomes. Here we show how to define a multitude of photodetector figures of merit in terms of a given POVM. We distinguish classical and quantum figures of merit and issue a conjecture regarding trade-off relations between them. We discuss the relationship between POVM elements and photodetector clicks, and how models of photodetectors may be tested by measuring either POVM elements or figures of merit. Finally, the POVM is advertised as a platform-independent way of comparing different types of photodetectors, since any such POVM refers to the Hilbert space of the incoming light, and not to any Hilbert space internal to the detector.
\end{abstract}

\section*{1. Introduction}

The most general mathematical description of a measurement on a quantum system is in terms of a positiveoperator valued measure (POVM). This is a set-which may be finite or infinite-of hermitian operators $\left\{\hat{\Pi}_{k}\right\}$ with nonnegative eigenvalues and with $\sum_{k} \hat{\Pi}_{k}=1$, where $k$ labels the different measurement outcomes and $\mathbb{1}$ is the identity operator on the Hilbert space associated with the system [1]. The POVM plays a central role in modern quantum information theory. For example, in order to assess how much information an eavesdropper on a quantum cryptographic protocol could have obtained, one has to take into account the most general possible measurement she could have performed [2]. Furthermore, various different forms of quantum tomography have been developed recently for experimentally determining individual POVM elements $\hat{\Pi}_{k}$. Detector tomography [3-9], self-consistent tomography [10], and SPAM tomography [11-13] use different sorts of assumptions to estimate from experimental data what POVM element corresponds to a given outcome of a quantum measurement.

The question considered here is how traditional figures of merit of a photodetector, such as bandwidth, dark-count rate, efficiency, jitter time, response time, spectral sensitivity, and photon-number resolution, are to be expressed in terms of the POVM pertaining to that detector ${ }^{1}$. This may not always be straightforward, as the Hilbert space associated with the radiation part of the electromagnetic field is overwhelmingly large: it is described by infinitely many modes, and each mode in turn is described by an infinite-dimensional Hilbert space. In practice, we often will restrict the Hilbert space of interest, e.g., to some finite wavelength range and some finite time window, but even so the Hilbert space may still be large.

The standard description of the radiation part of the electromagnetic field makes use of four mode numbers, which refer to properties of the classical mode functions, i.e., solutions to the classical source-free Maxwell equations. For example, the four numbers could describe (i) polarization and the three components of the wave

\footnotetext{
${ }^{1}$ Both the POVM and the figures-of-merit may be be functions of external parameters such as operating temperature, bias voltage, etc.
}
vector (corresponding to the standard expansion of the field in plane waves [14]), or (ii) energy, angular momentum, the $z$ component of angular momentum, and parity (corresponding to an expansion in multipole waves [14]), or (iii) energy and the $z$ components of momentum, orbital angular momentum, and spin angular momentum (corresponding to an expansion in Bessel waves [15] ${ }^{2}$ ).

The truly nonclassical quantum degree of freedom is encoded in the quantum state of the modes. For each mode $i$ (where the label $i$ is a shorthand notation for the four mode numbers) there is an infinite-dimensional Hilbert space which is spanned by the Fock states $|n\rangle_{i}$ where the nonnegative integer $n$ gives the number of photons in that mode. In correspondence with this distinction between classical and quantum degrees of freedom, we may divide traditional photodetector characteristics into two groups: one 'classical' group refers exclusively to physical properties of the classical mode functions of the detected light field. For example, spectral sensitivity, bandwidth, and jitter time all refer to the (classical) spectral degree of freedom, if we consider time as related to the spectrum by the Fourier transform. The second group of photodetector characteristics, which we may call the 'quantum' group, includes efficiency, dark count rate and photon-number resolution, all of which refer to photon-number statistics, as determined by expectation values of the annihilation and creation operators $\hat{a}_{i}$ and $\hat{a}_{i}^{+}$for each mode $i$, and products thereof ${ }^{3}$.

In addition to expressing individual photodetector characteristics in terms of the detector's POVM we may also wish to derive fundamental tradeoff relations between different characteristics. For example, if we increase the wavelength resolution of our detector, will that necessarily decrease the photon-number resolution? We may expect tradeoff relations to exist between characteristics that fall within the same group (jitter time and spectral sensitivity being an obvious example, resulting from time-frequency uncertainty relations), but prima facie not between characteristics that are in the two different groups. For example, photon-number resolution and spectral sensitivity are independent concepts and quantities, and while in practice tradeoff relations may arise from restrictions due to costs or operating temperature or the specific design of the detector, they do not, it seems, arise from fundamental laws of quantum physics. Indeed, we may write down a mathematically allowed POVM that suffers no such trade offs (see the Discussion and Conclusions section).

Some notational convention used throughout in this paper: $k$ will always be the integer index labeling measurement outcomes and the corresponding POVM elements $\hat{\Pi}_{k}$. Modes are indexed by integers $i$, whereas photon numbers are indicated by nonnegative integers $n$. Integers $j$ will be used to label finite-size bins in either the frequency or time domain.

\section*{2. Preliminaries}

\subsection*{2.1. Generalized measurements}

Standard textbook treatments of quantum measurement talk about an observable $\hat{O}$ as a hermitian or selfadjoint operator, with outcomes represented by the orthogonal set of its eigenvectors (eigenstates), and the corresponding eigenvalues giving the values of the physical quantity measured. But this description represents only a highly idealized subclass of all possible measurements, the so-called von Neumann measurements.

The modern, fully general, description of quantum measurements is in terms of a POVM. Here each outcome is represented by a positive hermitian operator $\hat{\Pi}_{k}$ with nonnegative real eigenvalues. (In the special case of an ideal measurement, $\hat{\Pi}_{k}$ would be the projector onto the eigenstate of the observable measured, say, $\hat{\Pi}_{k}=|k\rangle\langle k|$, and there is just a single nonzero eigenvalue for $\hat{\Pi}_{k}$, namely 1.) The label $k$ labels the outcome, and the probability of that outcome is determined by the quantum state $\hat{\rho}$ of the system on which the measurement is performed, through the Born rule
$$
\begin{equation*}
p_{k}=\operatorname{Tr}\left(\hat{\rho} \hat{\Pi}_{k}\right) . \tag{1}
\end{equation*}
$$

The condition $\sum_{k} p_{k}=1$ is entailed by completeness $\sum_{k} \hat{\Pi}_{k}=1$. Unlike for ideal measurements, different outcomes $k \neq k^{\prime}$ do not necessarily correspond to pairwise orthogonal projectors, and in general we can have
$$
\begin{equation*}
\operatorname{Tr}\left(\hat{\Pi}_{k} \hat{\Pi}_{k^{\prime}}\right) \neq 0 \text { for } k \neq k^{\prime} . \tag{2}
\end{equation*}
$$

This implies, for example, that a repeated measurement does not necessarily repeat the outcome.
Furthermore, measurement outcomes do not necessarily correspond to projectors onto pure states, and in general we can have
$$
\begin{equation*}
\operatorname{Tr}\left(\left(\hat{\Pi}_{k}\right)^{2}\right)<\left[\operatorname{Tr}\left(\hat{\Pi}_{k}\right)\right]^{2} . \tag{3}
\end{equation*}
$$

\footnotetext{
${ }^{2}$ Because of the transversality of radiation fields, there is a subtlety associated with the definition of spin and orbital angular momentum [14, 16], which, however, is of no concern here.
${ }^{3}$ The annihilation and creation operators act on photon-number states as $\hat{a}_{i}|n\rangle_{i}=\sqrt{n}|n-1\rangle_{i}$ and $\hat{a}_{i}^{+}|n\rangle_{i}=\sqrt{n+1}|n+1\rangle_{i}$.
}

We may even define a purity for the measurement outcome $k$ as
$$
\begin{equation*}
\operatorname{Pur}\left(\hat{\Pi}_{k}\right)=\frac{\operatorname{Tr}\left(\left(\hat{\Pi}_{k}\right)^{2}\right)}{\left[\operatorname{Tr}\left(\hat{\Pi}_{k}\right)\right]^{2}} . \tag{4}
\end{equation*}
$$

This definition is in full analogy to the purity of a quantum state, $\operatorname{Tr}\left(\hat{\rho}^{2}\right)$, which becomes even more clear if we define a unit-trace operator $\hat{\rho}_{k}=\hat{\Pi}_{k} /\left(\operatorname{Tr}\left(\hat{\Pi}_{k}\right)\right.$, so that $\operatorname{Pur}\left(\hat{\Pi}_{k}\right)=\operatorname{Tr}\left(\hat{\rho}_{k}^{2}\right)$. For an ideal von Neumann measurement this purity equals unity, which is the upper bound of the quantity on the right-hand side of (4). The lower bound on the purity is determined by the dimension $d$ of the Hilbert space:
$$
\begin{equation*}
\frac{1}{d} \leqslant \operatorname{Pur}\left(\hat{\Pi}_{k}\right) . \tag{5}
\end{equation*}
$$

This bound shows that the physical meaning of a POVM element $\hat{\Pi}_{k}$ not being pure is that there are multiple orthogonal input states that can lead to the same measurement outcome $k$. In fact, we may define an effective (not necessarily integer) Hilbert space dimension by
$$
\begin{equation*}
d_{\mathrm{eff}}(k)=\frac{1}{\operatorname{Pur}\left(\hat{\Pi}_{k}\right)}, \tag{6}
\end{equation*}
$$
which then counts how many (at least) such orthogonal states contribute to outcome $k$.
For an example of a POVM in optics, we need look no further than balanced heterodyne detection. The outcome of a heterodyne measurement consists of two real numbers, say $x$ and $y$, which are combined into a complex number $\alpha=x+\mathrm{i} y$. When $x$ and $y$ have been properly normalized, the measurement is represented by the POVM $\left\{\frac{1}{\pi}|\alpha\rangle\langle\alpha|\right\}^{4}$, where the state $|\alpha\rangle$ of the radiation field is a coherent state with amplitude $\alpha$. Coherent states are pure (and so this POVM is pure), but no coherent state is orthogonal to any other coherent state. (As this example shows, the effect of interference of external fields with the signal field prior to detecting photo currents is included in this POVM; similarly, spatial and spectral filtering applied to the signal before counting photons should be included in the POVM description of the detection process as a whole, too [17].)

Another example of a POVM in optics is the measurement of optical phase. Even though no hermitian operator exists that would have all desired canonical properties corresponding to the phase 'observable,' there is no problem defining the canonical POVM for optical phase [18].

For examples of how to include in the POVM the effects of standard forms of noise accompanying the photo detection process, dark counts and finite efficiency, see [19-26].

\subsection*{2.2. POVM elements and clicks}

The measurement performed by a photodetector is described fully by its POVM. However, there is no one-toone correspondence between a single POVM element and a single 'click' of the detector. Rather, each POVM element corresponds to one measurement outcome, which refers to the (classical) result at the end of the entire measurement process. Hence, a single measurement outcome may consist of multiple clicks. For a simple example, suppose our detector has 2 pixels that each can click at most twice in the short duration the detector is switched on (e.g., because of dead time). The total number of different outcomes is then $3^{2}=9$, since each pixel may click 0,1 , or 2 times. Consequently, there are 9 POVM elements in this particular case.

A special measurement outcome is the null outcome, where no clicks are recorded at all. We will assume here and in all of the following that our detector is sensitive only to light in a small fraction of all possible modes, and hence that the no-click POVM element covers the overwhelmingly large part of the full Hilbert space. For that reason it is convenient to label the no-click outcome with $k=$ null and write the completeness relation as
$$
\begin{equation*}
\hat{\Pi}_{\mathrm{null}}=\mathbb{1}-\sum_{k=1}^{N} \hat{\Pi}_{k}=: \mathbb{1}-\hat{\Pi}, \tag{7}
\end{equation*}
$$
in terms of the NPOVM elements $\left\{\hat{\Pi}_{k}, k=1 \ldots N\right\}$ that correspond to nonzero numbers of detector clicks (so in the simple example, $N=8$ ). We are going to focus on the information obtained from the clicks and we will discard here the information one might possibly obtain from getting no click. That is, we assume we are interested in extracting information about photons that are actually present. The null outcome simply and merely means we failed to detect the photon(s). To see why the null outcome carries very little information about what photon(s) may be present, suppose there are very many, say $K \gg 1$ modes not detectable, and a much smaller number, $L \ll K$, that are detectable. Before turning on our detector we lack $\log _{2}(K+L)$ bits of information about what photon may be present, and after getting the null outcome we have reduced this missing information to (at most) $\log _{2}(K)$ bits. We thus merely gained (at most) $\log _{2}(K+L)-\log _{2}(K) \approx L / K \ll 1$ bits.

\footnotetext{
${ }^{4}$ The normalization of the POVM, i.e., the prefactor $1 / \pi$, follows from $\int \mathrm{d} x \int \mathrm{~d} y|x+\mathrm{i} y\rangle\langle x+\mathrm{i} y|=\pi 1$.
}

The operator $\hat{\Pi}$ defined in (7) represents all detector clicks together. Some of these clicks may be dark counts, not caused by any photon; we do take dark counts into account, see section 3.4.

\subsection*{2.3. Discrete modes versus continuum fields}

Since we will only consider the spectral/temporal degree of freedom among the 4 classical degrees of freedom light possesses, in all explicit examples we need just a single mode number, either frequency $\omega$ or time $t$. Both of these are continuous quantities, but Hilbert spaces occurring in quantum mechanics are always taken to be separable, i.e., have a countable basis. We can go from a continuum description of frequency to a discrete set of modes by a method explained in [27]. Here is a summary ${ }^{5}$.

First, we can define a discrete orthonormal set of (complex) mode functions $\left\{\phi_{i}(\omega)\right\}$ normalized such that
$$
\begin{equation*}
\int_{0}^{\infty} \mathrm{d} \omega \phi_{i}(\omega) \phi_{j}^{*}(\omega)=\delta_{i j} . \tag{8}
\end{equation*}
$$

Then we can define a creation and annihilation operator for each discrete mode $i$ by
$$
\begin{align*}
\hat{a}_{i}^{+} & =\int_{0}^{\infty} \mathrm{d} \omega \phi_{i}(\omega) \hat{a}^{+}(\omega) \\
\hat{a}_{i} & =\int_{0}^{\infty} \mathrm{d} \omega \phi_{i}^{*}(\omega) \hat{a}(\omega) \tag{9}
\end{align*}
$$
where $\hat{a}^{+}(\omega)$ and $\hat{a}(\omega)$ are the standard creation and annihilation operators for photons with frequency $\omega$. The definition (9) is such that the commutation rule $\left[\hat{a}_{i}, \hat{a}_{i^{\prime}}^{+}\right]=\delta_{i i^{\prime}}$ follows from $\left[\hat{a}(\omega), \hat{a}^{+}\left(\omega^{\prime}\right)\right]=\delta\left(\omega-\omega^{\prime}\right)$ and equation (8) ${ }^{6}$. The integration range here and in equation (8) extends over just the positive frequencies, since creation operators $\hat{a}^{+}(\omega)$ are defined only for those frequencies. A pure single-photon state containing exactly one photon in mode $i$ is defined as
$$
\begin{equation*}
\left|\phi_{i}\right\rangle=\int_{0}^{\infty} \mathrm{d} \omega \phi_{i}(\omega) \hat{a}^{+}(\omega)|\mathrm{vac}\rangle=\hat{a}_{i}^{+}|\mathrm{vac}\rangle, \tag{10}
\end{equation*}
$$
with |vac the vacuum state containing no photons. States of multiple photons can be described in terms of the operator $\hat{a}_{i}^{+}$, too. For example, the state with exactly $n$ photons in mode $i$ (and none elsewhere) is
$$
\begin{equation*}
|n\rangle_{i}=\frac{\left(\hat{a}_{i}^{+}\right)^{n}}{\sqrt{n!}}|\mathrm{vac}\rangle, \tag{11}
\end{equation*}
$$
and a coherent state of mode $i$ with complex amplitude $\alpha$ is
$$
\begin{align*}
|\alpha\rangle_{i} & =\exp \left(\alpha \hat{a}_{i}^{+}-\alpha^{*} \hat{a}_{i}\right)|\operatorname{vac}\rangle \\
& =\exp \left(-|\alpha|^{2} / 2\right) \sum_{n} \frac{\alpha^{n}}{\sqrt{n!}}|n\rangle_{i} . \tag{12}
\end{align*}
$$

For an ideal 100\% efficient detector that detects the presence of photons (not their energy), the probability $\mathrm{d} P_{i}(t)$ to detect a photon in the state $\left|\phi_{i}\right\rangle$ during a small time interval between $t$ and $t+\mathrm{d} t$ would be
$$
\begin{equation*}
\mathrm{d} P_{i}(t)=\frac{\mathrm{d} t}{2 \pi}\left|\int_{0}^{\infty} \mathrm{d} \omega \phi_{i}(\omega) \exp (-\mathrm{i} \omega t)\right|^{2}, \tag{13}
\end{equation*}
$$
which is such that the ideal single-photon detection rate, $P_{i}(t)$, integrates to 1 :
$$
\begin{equation*}
\int_{-\infty}^{\infty} \mathrm{d} t P_{i}(t)=\int_{0}^{\infty} \mathrm{d} \omega\left|\phi_{i}(\omega)\right|^{2}=1 \tag{14}
\end{equation*}
$$

\section*{3. Detector characteristics in terms of POVMs}

In the first two subsections to follow we will discuss the information gained about a single photon from a single click of the detector. Subsequent subsections discuss what sort of information is obtained about the presence of multiple photons and what information is obtained from multiple clicks.

\footnotetext{
${ }^{5}$ Unlike [27] we avoid defining an operator $\hat{a}^{+}(t)$ here, and do not need to artificially extend the integration over $\omega$ to include negative frequencies. The (im) possibility of localizing a photon is an interesting theoretical issue in this context, but in practice we may safely leave this can of worms closed; see [28] for an extensive discussion.
${ }^{6}$ If in addition we assume completeness in $L^{2}[0, \infty]$ of the set of functions $\left\{\phi_{i}(\omega)\right\}$, then we can expand $\hat{a}^{+}(\omega)=\sum_{i} \phi_{i}^{*}(\omega) \hat{a}_{i}^{+}$.
}

\subsection*{3.1. Single-photon bandwidth}

First let us define the single-photon part of an arbitrary POVM element,
$$
\begin{equation*}
\hat{\Pi}_{k}^{(1)}=\hat{P}^{(1)} \hat{\Pi}_{k} \hat{P}^{(1)} . \tag{15}
\end{equation*}
$$

Here $\hat{P}^{(1)}$ is the projector onto the (relevant part of) the 1 -photon subspace, defined by a sum over all modes $i$ (with $\left|\phi_{i}\right\rangle$ defined in (10))
$$
\begin{equation*}
\hat{P}^{(1)}=\sum_{i}\left|\phi_{i}\right\rangle\left\langle\phi_{i}\right| . \tag{16}
\end{equation*}
$$

We can now define two bandwidth-related quantities. First,
$$
\begin{equation*}
\Omega_{k}^{(1)}=\operatorname{Tr}\left(\hat{\Pi}_{k}^{(1)}\right) \tag{17}
\end{equation*}
$$
is the effective size of the single-photon Hilbert space covered by outcome $k$. For example, suppose outcome $k$ represents the click of one particular pixel $k$ that is sensitive only to one particular wavelength. If that pixel detects a photon with that wavelength with probability $p<1$, then we would have $\Omega_{k}^{(1)}=p<1$. Normally, one expects sensitivity to a range of wavelengths such that $\Omega_{k}^{(1)}>1$ or even $\Omega_{k}^{(1)} \gg 1$. The second quantity we define is
$$
\begin{equation*}
\Omega^{(1)}=\sum_{k=1}^{N} \Omega_{k}^{(1)}=\operatorname{Tr}\left(\hat{P}^{(1)} \hat{\Pi} \hat{P}^{(1)}\right), \tag{18}
\end{equation*}
$$
which is a measure of the effective size of the single-photon Hilbert space covered by all of the detector's possible clicks. Note that these definitions of bandwidth are all basis-independent and dimensionless, and in particular do not distinguish between spectral and temporal degrees of freedom. The bandwidth thus defined is appropriate in a communication context, in which $\Omega^{(1)}$ would roughly be the total number of single-photon channels that could be detected (but not necessarily distinguished).

For an arbitrary single-photon POVM element we can find its diagonal form
$$
\begin{equation*}
\hat{\Pi}_{k}^{(1)}=\sum_{i} w_{i}^{(k)}\left|\phi_{i}^{(k)}\right\rangle\left\langle\phi_{i}^{(k)}\right|, \tag{19}
\end{equation*}
$$
where $\left|\phi_{i}^{(k)}\right\rangle$ denotes a pure single-photon state of mode $i$ (see equation (10)). Note that each POVM element $\hat{\Pi}_{k}^{(1)}$ may be diagonal in a different basis $\left\{\left|\phi_{i}^{(k)}\right\rangle\right\}$, hence the superscript $(k)$. The weight $w_{i}^{(k)}$ has the meaning of the conditional probability of getting measurement outcome $k$ given an input state $\left|\phi_{i}^{(k)}\right\rangle$,
$$
\begin{equation*}
\operatorname{Pr}(k \mid i)=\operatorname{Tr}\left(\hat{\Pi}_{k}^{(1)}\left|\phi_{i}^{(k)}\right\rangle\left\langle\phi_{i}^{(k)}\right|\right)=\left\langle\phi_{i}^{(k)}\right| \hat{\Pi}_{k}^{(1)}\left|\phi_{i}^{(k)}\right\rangle=w_{i}^{(k)}, \tag{20}
\end{equation*}
$$
and so it necessarily lies between 0 and 1 . Moreover, their sum over $i$ gives the bandwidth (17): $\sum_{i} w_{i}^{(k)}=\Omega_{k}^{(1)}$.
What does, conversely, the measurement outcome $k$ imply about the input state of the photon that was just detected? Suppose the input states $\left|\phi_{i}^{(k)}\right\rangle$ for different $i$ appear a priori with some probability $\operatorname{Pr}(i)$. After getting outcome $k$ we can update our probability distribution over input states $i$ to
$$
\begin{equation*}
\operatorname{Pr}(i \mid k)=\frac{w_{i}^{(k)} \operatorname{Pr}(i)}{\operatorname{Pr}(k)}, \tag{21}
\end{equation*}
$$
with $\operatorname{Pr}(k)=\sum_{i} \operatorname{Pr}(i) w_{i}^{(k)}$ the a priori probability to get outcome $k$. We can quantify the amount of information we still lack about which mode the photon was in by the Shannon entropy
$$
\begin{equation*}
H^{(k)}=-\sum_{i} \operatorname{Pr}(i \mid k) \log _{2} \operatorname{Pr}(i \mid k) . \tag{22}
\end{equation*}
$$

This quantity depends on our prior knowledge (or prior assumptions) about the input states. We can eliminate this dependence by making $\operatorname{Pr}(i) / \operatorname{Pr}(k)$ independent of $i$. In that case we would have $\operatorname{Pr}(i \mid k)=w_{i}^{(k)} / \Omega_{k}^{(1)}$, and the corresponding Shannon entropy is then an effectively input-independent quantity-and we will use a calligraphic script to emphasize this useful property-that characterizes the detector,
$$
\begin{equation*}
\mathcal{H}^{(k)}=-\sum_{i} \frac{w_{i}^{(k)}}{\Omega_{k}^{(1)}} \log _{2} \frac{w_{i}^{(k)}}{\Omega_{k}^{(1)}} . \tag{23}
\end{equation*}
$$

In terms of $\hat{\rho}_{k}=\hat{\Pi}_{k}^{(1)} / \operatorname{Tr}\left(\hat{\Pi}_{k}^{(1)}\right)$, we get the manifestly input-independent
$$
\begin{equation*}
\mathcal{H}^{(k)}=-\operatorname{Tr}\left[\hat{\rho}_{k} \log _{2} \hat{\rho}_{k}\right] . \tag{24}
\end{equation*}
$$

We could also use the so-called collision entropy (which is the Renyi entropy $H_{\alpha}$ of order $\alpha=2$ ) to quantify our lack of knowledge as
$$
\begin{equation*}
H_{\alpha=2}^{(k)}=-\log _{2}\left(\sum_{i} \operatorname{Pr}(i \mid k)^{2}\right) . \tag{25}
\end{equation*}
$$

Again, when we assume $\operatorname{Pr}(i) / \operatorname{Pr}(k)$ is independent of $i$, this collision entropy becomes input-independent, and in fact we get
$$
\begin{equation*}
\mathcal{H}_{\alpha=2}^{(k)}=-\log _{2}\left(\operatorname{Pur}\left(\hat{\Pi}_{k}^{(1)}\right)\right), \tag{26}
\end{equation*}
$$
with the purity Pur (.) defined in (4).
And so the purity of $\hat{\Pi}_{k}^{(1)}$ and the Shannon entropy quantify in different but well-defined ways the lack of specificity of the outcome $k$. In the following subsections we use these same ideas to define spectral and timing resolution, as well as the photon-number resolving capabilities of a given detector. We choose to utilize the Shannon entropy there, but could use the collision entropy or the purity just as well.

\subsection*{3.2. Wavelength and timing resolution}

Given the diagonal form (19) of $\hat{\Pi}_{k}^{(1)}$ we find the normalized a posteriori probability distribution over $\omega$ that outcome $k$ implies as
$$
\begin{equation*}
\operatorname{Pr}(\omega \mid k)=\sum_{i} \operatorname{Pr}(i \mid k)\left|\phi_{i}^{(k)}(\omega)\right|^{2} . \tag{27}
\end{equation*}
$$

Analogously, we can define an a posteriori probability distribution over detection times of the photon as
$$
\begin{equation*}
\operatorname{Pr}(t \mid k) \mathrm{d} t=\sum_{i} \operatorname{Pr}(i \mid k) \mathrm{d} P_{i}(t), \tag{28}
\end{equation*}
$$
with $\mathrm{d} P_{i}(t)$ defined in (13). These probability distributions are over continuous quantities. In practice finite precision forces one to bin the frequency and time measurements into finite-sized intervals. So, let us first divide the frequency range into equal-sized ${ }^{7}$ small frequency intervals $\delta \omega$. Then, given the probability distributions $\operatorname{Pr}(\omega \mid k)$ for each POVM element $\hat{\Pi}_{k}^{(1)}$, we may define for each positive integer $j$ the probability
$$
\begin{equation*}
p(j \mid k)=\int_{(j-1) \delta \omega}^{j \delta \omega} \mathrm{~d} \omega \operatorname{Pr}(\omega \mid k), \tag{29}
\end{equation*}
$$
which is the a posteriori probability for the detected photon to belong to frequency bin $j$. The Shannon entropy
$$
\begin{equation*}
H_{\omega}^{(k)}=-\sum_{j} p(j \mid k) \log _{2} p(j \mid k), \tag{30}
\end{equation*}
$$
properly quantifies the amount of information (in units of bits) we still lack after having obtained outcome $k$ about which frequency interval the photon we just detected belongs to.

We can define the analogous probabilities $q(j \mid k)$ for integers $j$ (not necessarily positive) and the corresponding entropy $H_{t}^{(k)}$ for the lack of information about the photon's time of detection once we have divided time in bins of finite size $\delta t$, as
$$
\begin{align*}
q(j \mid k) & =\int_{(j-1) \delta t}^{j \delta t} \mathrm{~d} t \operatorname{Pr}(t \mid k) \\
H_{t}^{(k)} & =-\sum_{j} q(j \mid k) \log _{2} q(j \mid k) \tag{31}
\end{align*}
$$

For a given POVM we can take the weighted averages of the individual entropies
$$
\begin{equation*}
\bar{H}_{\omega, t}=\sum_{k=1}^{N} \frac{\Omega_{k}}{\Omega} H_{\omega, t}^{(k)}, \tag{32}
\end{equation*}
$$
as a measure of how much information about frequency or time we anticipate lacking on average.
For each outcome $k$, we have the entropic uncertainty relation $[29]^{8}$
$$
\begin{equation*}
H_{\omega}^{(k)}+H_{t}^{(k)}>\log _{2}(e)-1-\log _{2} \frac{\delta \omega \delta t}{2 \pi} . \tag{33}
\end{equation*}
$$

The two weighted averages satisfy the same uncertainty relation, since the right-hand side of (33) is independent of $k$, i.e., we also have
$$
\begin{equation*}
\bar{H}_{\omega}+\bar{H}_{t}>\log _{2}(e)-1-\log _{2} \frac{\delta \omega \delta t}{2 \pi} . \tag{34}
\end{equation*}
$$

\footnotetext{
${ }^{7}$ We could drop the assumption of equal-sized frequency intervals and switch to, say, equal-sized intervals in wavelength.
${ }^{8}$ Uncertainty relations for the collision entropy, even including binning, as well as other Renyi entropies, can be found in [30].
}

The entropies we have defined here do depend on our choices of $\delta \omega$ and $\delta t$. The smaller we pick our bin sizes, the larger will be the missing information. Roughly speaking, each time we make the interval smaller by a factor of 2 , $\bar{H}$ increases by approximately 1 bit (in fact, by at most 1 bit). This strong dependence on bin size is not a desirable property, even though we can still make sensible comparisons between different detectors for given values of $\delta \omega$ and $\delta t$. To obtain a more useful (and dimensionful) quantity we could adopt the following convention. Pick interval sizes $\delta \omega$ and $\delta t$ such that the averaged missing information $\bar{H}_{\omega, t}$ equals a few bits. That is, these intervals are really too small to be resolved by the detector. If we define
$$
\begin{align*}
\Delta \omega & =2^{\bar{H}_{\omega}} \delta \omega, \\
\Delta t & =2^{\bar{H}_{t}} \delta t \tag{35}
\end{align*}
$$
then the quantities on the left-hand side, $\Delta \omega$ and $\Delta t$, satisfy an uncertainty relation independent of the bin sizes $\delta \omega$ and $\delta t$,
$$
\begin{equation*}
\Delta \omega \Delta t \geqslant e \pi \approx 8.54 . \tag{36}
\end{equation*}
$$

We could take $\Delta \omega$ and $\Delta t$ as measures of the average frequency and timing resolutions of our detector, respectively, even though they still weakly depend on the bin sizes. (We could simply require the entropies to equal, say, 4 bits, in order to fix the bin sizes and make the definitions for $\Delta \omega$ and $\Delta t$ unique.)

\subsection*{3.3. Photon number resolution}

In order to talk about photon-number resolution we do need to first distinguish between different modes. Let us fix one mode of interest, $i$, so that the Hilbert space of interest is spanned by the Fock states $|n\rangle_{i}$. Then, for a given outcome $k$ we need the a posteriori probability distribution over different numbers of photons in mode $i$ that outcome $k$ implies. We may write this a posteriori probability as a conditional probability $\operatorname{Pr}(n \mid k, i)$ : given outcome $k$ and given an input mode $i$, what is the probability for a number of photons equal to $n$ ? The entropy
$$
\begin{equation*}
H_{n, i}^{(k)}=-\sum_{n} \operatorname{Pr}(n \mid k ; i) \log _{2} \operatorname{Pr}(n \mid k ; i) \tag{37}
\end{equation*}
$$
quantifies (in bits) how much information concerning the photon number $n$ in mode $i$ is still missing after we have obtained outcome $k$.

Again we could assume, for the purpose of an input-independent definition, that the a priori probabilities of different numbers of photons are equal over some finite range. If we define weights
$$
\begin{equation*}
\Omega_{k, i}^{(n)}=\langle\operatorname{vac}|\left(\hat{a}_{i}\right)^{n} \hat{\Pi}_{k}\left(\hat{a}_{i}^{+}\right)^{n}|\operatorname{vac}\rangle / n!, \tag{38}
\end{equation*}
$$
the sought-after a posteriori probability distribution over $n$ is then given by
$$
\begin{equation*}
\operatorname{Pr}(n \mid k, i)=\frac{\Omega_{k, i}^{(n)}}{W_{k, i}}, \tag{39}
\end{equation*}
$$
where $W_{k, i}=\sum_{n} \Omega_{k, i}^{(n)}$. The Shannon entropy may be written then as
$$
\begin{equation*}
\mathcal{H}_{n, i}^{(k)}=-\sum_{n} \frac{\Omega_{k, i}^{(n)}}{W_{k, i}} \log _{2} \frac{\Omega_{k, i}^{(n)}}{W_{k, i}} . \tag{40}
\end{equation*}
$$
(And again we may average this quantity over all POVM elements by summing either over $k$ or over $i$ or over both, and giving each term a relative weight $W_{k, i}$.)

The sums over $n$ here all extend in principle over the entire range of allowed $n$ but in practice these sums really contain only a few non-neglible terms. For instance, if we have an array of highly efficient on/off detectors and two of them click, then the probability that more than a dozen of photons (in the right wavelength range) caused just these two clicks is negligible.

It is really simpler here to use the purity to quantity photon-number resolution. We first restrict the POVM element $\hat{\Pi}_{k}$ to mode $i$,
$$
\begin{equation*}
\hat{\Pi}_{k}^{(i)}=\sum_{n}|n\rangle_{i}\langle n| \hat{\Pi}_{k}|n\rangle_{i}\langle n|, \tag{41}
\end{equation*}
$$
and then get the collision entropy as
$$
\begin{equation*}
\mathcal{H}_{k}^{(i)}=-\log _{2} \operatorname{Pur}\left(\hat{\Pi}_{k}^{(i)}\right), \tag{42}
\end{equation*}
$$
as a measure for how specific outcome $k$ is about the number of photons in mode $i$ that caused it.
Note, finally, that there is an entropic uncertainty relation for photon number and phase like that between time and frequency. However, phase sensitivity is usually not considered a property of the detector, but rather of the interferometric setup in which the photodetector is placed. That is why we will not consider phase sensitivity here.

\subsection*{3.4. Efficiency and dark count rates}

Efficiency is defined as the probability that a single photon (in a given mode) is detected. Clearly, the efficiency is in general a mode-dependent quantity. We can diagonalize not just each individual single-photon detection POVM element (as we did in equation (19)), but their sum
$$
\begin{equation*}
\sum_{k=1}^{N} \hat{\Pi}_{k}^{(1)}=\hat{P}^{(1)} \hat{\Pi} \hat{P}^{(1)}=\sum_{i} w_{i}\left|\phi_{i}\right\rangle\left\langle\phi_{i}\right| . \tag{43}
\end{equation*}
$$

The weights $w_{i}$ appearing here are really efficiencies for the modes $i: w_{i}$ is the probability that a single photon present in mode $i$ causes a click. And so we can simply define efficiencies $\eta_{i}=w_{i}$. The largest of all $w_{i}$ s is the efficiency of the detector at its most sensitive point,
$$
\begin{equation*}
\eta_{\max }=\max _{i} w_{i} . \tag{44}
\end{equation*}
$$

If we are interested in a particular mode $i^{\prime}$ that is not a basis vector in the basis that diagonalizes $\sum_{k=1}^{N} \hat{\Pi}_{k}^{(1)}$, then we can still define the appropriate single-photon detection efficiency as
$$
\begin{equation*}
\eta_{i^{\prime}}=\left\langle\phi_{i^{\prime}}\right| \hat{\Pi}\left|\phi_{i^{\prime}}\right\rangle=\langle\operatorname{vac}| a_{i^{\prime}} \hat{\Pi} a_{i^{\prime}}^{+}|\operatorname{vac}\rangle . \tag{45}
\end{equation*}
$$

Dark count rates are determined by the probabilities of detector clicks when no photon is present. So, we clearly need the quantities
$$
\begin{equation*}
d_{k}=\operatorname{Tr}\left(\hat{P}^{(0)} \hat{\Pi}_{k} \hat{P}^{(0)}\right)=\langle\operatorname{vac}| \hat{\Pi}_{k}|\operatorname{vac}\rangle=\Omega_{k}^{(0)} . \tag{46}
\end{equation*}
$$

The dark-count rate is not mode dependent, but it may depend on $k$. The dark count rate for a detector that is switched on for a duration $T$ is in fact
$$
\begin{equation*}
d=\frac{\sum_{k=1}^{N} d_{k} N(k)}{T}, \tag{47}
\end{equation*}
$$
because the numerator equals the expected number of dark counts provided we let $N(k)$ denote the total number of clicks occurring in outcome $k$ (recall section 2.2).

\subsection*{3.5. Response time and detection rate}

Maximum rate and dead time or response time are determined by correlations in time between multiple clicks. The response time may depend on the mode detected. Consider a single-photon state of the form
$$
\begin{equation*}
|\phi\rangle=\int_{0}^{\infty} \mathrm{d} \omega \phi(\omega) \hat{a}^{+}(\omega)|\mathrm{vac}\rangle \tag{48}
\end{equation*}
$$
and consider a time translated version of this state (i.e., the state that would result from free evolution of $|\phi\rangle$ over some time $\tau>0$ )
$$
\begin{equation*}
\left|\phi_{\tau}\right\rangle=\hat{T}(\tau)|\phi\rangle=: \int_{0}^{\infty} \mathrm{d} \omega \phi(\omega) \exp (-\mathrm{i} \omega \tau) \hat{a}^{+}(\omega)|\mathrm{vac}\rangle . \tag{49}
\end{equation*}
$$

If $\tau$ is not much larger than the response time we expect this quantity:
$$
\begin{equation*}
P(0, \tau):=\left\langle\phi, \phi_{\tau}\right| \hat{\Pi}\left|\phi, \phi_{\tau}\right\rangle, \tag{50}
\end{equation*}
$$
i.e., the joint probability to detect both a photon in mode $\phi$ and one in the time-translated mode $\phi_{\tau}$, to be less than the product of the two individual single-photon detection probabilities
$$
\begin{align*}
& P(0):=\langle\phi| \hat{\Pi}|\phi\rangle, \\
& P(\tau):=\left\langle\phi_{\tau}\right| \hat{\Pi}\left|\phi_{\tau}\right\rangle . \tag{51}
\end{align*}
$$

If, for a given mode, we define the response time as the time it takes to go from $10 \%$ to $90 \%$ of maximum detection probability (which we assume is associated with detection at $\tau=0$ ), then we need the time delays $\tau_{10}$ and $\tau_{90}$ such that
$$
\begin{align*}
& P\left(0, \tau_{10}\right)=\frac{1}{10} P(0)^{2}, \\
& P\left(0, \tau_{90}\right)=\frac{9}{10} P(0)^{2} . \tag{52}
\end{align*}
$$

This assumes $\tau_{90}>\tau_{10}$, and if there is no such $\tau_{90}$ satisfying the above requirement for a given $\tau_{10}$, then we can set $\tau_{90}=\infty$. The response time for a particular mode $\phi_{i}$ is then defined as $\theta_{i}=\tau_{90}-\tau_{10}$. A total detection rate can then be defined as a sum of inverse response times $\theta_{i}^{-1}$ for the modes $i$ that diagonalize $\sum_{k=1}^{N} \hat{\Pi}_{k}^{(1)}$,
$$
\begin{equation*}
R=\sum_{i} \theta_{i}^{-1} . \tag{53}
\end{equation*}
$$

This quantity automatically takes into account the possibility of having a large array of parallel detectors. Even if each of the detectors in the array has a slow response, the rate $R$ may still be high.

\section*{4. Discussion and conclusions}

We have shown here how standard photo detector figures of merit can be directly expressed in terms of the POVM describing the quantum properties of the photo detection. Since the POVM is fully quantummechanical, so are the figures of merit thus defined.

One advantage shared by the standard figures of merit and the POVM is that they do not refer to the Hilbert spaces internal to the photodetector (the Hilbert spaces associated with phonons, excitons, polaritons, discrete energy levels of single absorbers, etc, etc), but only to the Hilbert space and properties of the photons that are being detected. The advantage of the POVM over the figures of merit is that it, in principle, contains all information about how the photodetector's clicks provide information about the incoming photons.

A quantum field theory description of light shows that the standard detector characteristics fall into two groups: one group refers to the classical degrees of freedom of the classical mode functions, the other to the quantum degree of freedom related to photon statistics. The conjecture is that there are no fundamental-as opposed to practical—tradeoff relations between characteristics from the different groups. To illustrate this, consider the following POVM, perfectly legitimate from the mathematical point of view. For a given set of orthogonal modes $\left\{\phi_{i}(\omega)\right\}$, define
$$
\begin{equation*}
\hat{\Pi}_{n}^{i}=|n\rangle_{i}\langle n|, \tag{54}
\end{equation*}
$$
with $|n\rangle_{i}$ the state of exactly $n$ photons in mode $i$ (and no photons in any other mode). Every mode $i$ must satisfy a time-frequency uncertainty relation for its mode function (and we gave such relations in two different forms in section 3.2), but the measurement is perfectly number-resolving, and has zero dark counts irrespective of the choice of basis $\left\{\phi_{i}(\omega)\right\}$.

Photon-number resolution and spectral and timing resolution were all defined here in terms of entropic quantities. The latter quantify the amount of information still missing (about photon number, wavelength, and time of arrival, respectively, of the input light) after we have obtained a particular measurement outcome. These entropic quantities are all dimensionless, but we also showed how dimensionful quantities like bandwidth (in Hz ) or timing resolution (in seconds) may be obtained from the entropic quantities.

Finally, the purity of a POVM element seems a useful additional (nontraditional) figure-of-merit: roughly speaking, it quantifies how many different orthogonal quantum input states could lead to exactly the same measurement outcome.

The strategy followed in this paper was to consider the POVM as given. The next questions to be considered are (i) how one obtains such a POVM description, and (ii) how to experimentally test it. The photo-detection problem in all its generality is too complicated to allow for an $a b$ initio solution, and one will have to resort to simplified physical models that can, at a minimum, be used to fit to data. Model selection [31] is then a nice statistical technique that allows one to rank different models, based on how well the models fit the data and how many fitting parameters they use. This technique is especially useful for reducing the number of parameters if the relevant Hilbert space is large [32, 33], as it indeed is for the photo-detection problem.

One way to get relevant test data is to perform small-scale detector tomography (large-scale tomography is not feasible). That is, by restricting oneself to a not-too-large Hilbert space (spanned by, say, the states with at most, say, 20 photons [4] in 1 or 2 modes, as detected by one particular pixel), one may experimentally estimate the corresponding POVM elements. These estimates can be used directly to evaluate one's models. While tradeoff relations obtained within such a model may not constitute the fundamental limits of photodetection, they should be of great practical interest nonetheless.

The other way to test model descriptions, is by measuring figures-of-merit like quantum efficiency as a function of, say, wavelength (at different temperatures or while varying other control parameters) and comparing the result to what the underlying theory says about this functional dependence (either directly or indirectly via the POVM). This approach has been successfully adopted in several recent experiments on nanowire superconducting single-photon detectors [34-38].

\section*{Acknowledgments}

I thank Michael Raymer and Andrzej Veitia, as well as the participants in the DARPA DSO Detect Theory Kickoff and Technical Exchange Meeting for useful discussions.

This work is supported by funding from DARPA under Contract No. W911NF-17-1-0267.

\section*{References}
[1] Kraus K 1983 States, Effects and Operations (Lecture Notes in Physics vol 190) (Berlin: Springer)
[2] Scarani V,Bechmann-Pasquinucci H, Cerf N J, Dušek M, Lütkenhaus N and Peev M 2009 Rev. Mod. Phys. 811301
[3] Luis A and Sánchez-Soto L L 1999 Phys. Rev. Lett. 833573
[4] Feito A, Lundeen J, Coldenstrodt-Ronge H, Eisert J, Plenio M and Walmsley I 2009 New J. Phys. 11093038
[5] Lundeen J, Feito A, Coldenstrodt-Ronge H, Pregnell K, Silberhorn C, Ralph T, Eisert J, Plenio M and Walmsley I 2009 Nat. Phys. 527
[6] Zhang L, Datta A, Coldenstrodt-Ronge H B, Jin X-M, Eisert J, Plenio M B and Walmsley I A 2012 New J. Phys. 14115005
[7] Natarajan C M, Zhang L, Coldenstrodt-Ronge H, Donati G, Dorenbos S N, Zwiller V, Walmsley I A and Hadfield R H 2013 Opt. Express 21893
[8] Cooper M, Karpiński M and Smith B J 2014 Nat. Commun. 54332
[9] Humphreys P C, Metcalf B J, Gerrits T, Hiemstra T, Lita A E, Nunn J, Nam S W, Datta A, Kolthammer W S and Walmsley I A 2015 New J. Phys. 17103044
[10] Mogilevtsev D, Reháček J and Hradil Z 2012 New J. Phys. 14095001
[11] Stark C 2014 Phys. Rev. A 89052109
[12] Jackson C and van Enk S J 2015 Phys. Rev. A 92042312
[13] McCormick A F, van Enk S J and Beck M 2017 Phys. Rev. A 95042329
[14] Cohen-Tannoudji C, Dupont-Roc J and Grynberg G 2001 Photons and Atoms: Introduction to Quantum Electrodynamics (New York: Wiley)
[15] van Enk S J and Nienhuis G 1994 J. Mod. Opt. 41963
[16] Simmons J W and Guttmann M J 1970 States, Waves, and Photons: A Modern Introduction to Light (Reading, MA: Addison-Wesley)
[17] van Enk S J 2017 Phys. Rev. A 96033834
[18] Hall M W 1991 Quantum Opt.: J. Eur. Opt. Soc. B 37
[19] Barnett S M, Phillips L S and Pegg D T 1998 Opt. Commun. 15845
[20] Lee H, Yurtsever U, Kok P, Hockney G M, Adami C, Braunstein S L and Dowling J P 2004 J. Mod. Opt. 511517
[21] Tyc T and Sanders B C 2004 J. Phys. A: Math. Gen. 377341
[22] Semenov A, Turchin A and Gomonay H 2008 Phys. Rev. A 78055803
[23] Afek I, Natan A, Ambar O and Silberberg Y 2009 Phys. Rev. A 79043830
[24] Audenaert K M and Scheel S 2009 New J. Phys. 11113052
[25] Sperling J, Vogel W and Agarwal G 2012 Phys. Rev. A 85023820
[26] Miroshnichenko G P and Trifanov A 2013 Eur. Phys. J. D 671
[27] Blow K, Loudon R, Phoenix S J and Shepherd T 1990 Phys. Rev. A 424102
[28] Keller O 2005 Phys. Rep. 4111
[29] Białynicki-Birula I and Mycielski J 1975 Commun. Math. Phys. 44129
[30] Bialynicki-Birula I 2006 Phys. Rev. A 74052101
[31] Burnham K P and Anderson D R 2003 Model Selection and Multimodel Inference: A Practical Information-Theoretic Approach (New York: Springer)
[32] Schwarz L and van Enk S J 2013 Phys. Rev. A 88032318
[33] Ferrie C 2014 New J. Phys. 16093035
[34] Renema J, Frucci G, Zhou Z, Mattioli F, Gaggero A, Leoni R, De Dood M, Fiore A and Van Exter M 2012 Opt. Express 202806
[35] Renema J, Frucci G, Zhou Z, Mattioli F, Gaggero A, Leoni R, de Dood M J, Fiore A and van Exter M P 2013 Phys. Rev. B 87174526
[36] Renema J et al 2014 Phys. Rev. Lett. 112117604
[37] Wang Q, Renema J, Gaggero A, Mattioli F, Leoni R, van Exter M and de Dood M 2015 J. Appl. Phys. 118134501
[38] Gaudio Retal 2016 Appl. Phys. Lett. 109031101