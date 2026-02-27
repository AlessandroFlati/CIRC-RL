# Estensione Teorica: Normalizzazione delle Dinamiche di Transizione

## 1. Motivazione

Il framework CIRC-RL originale (Sezioni 3.1--3.6 di `CIRC-RL_Framework.md`) si
concentra sull'invarianza dei **meccanismi di ricompensa**.  Le Fasi 1 e 2
identificano il grafo causale, selezionano le feature che sono antenati causali
della ricompensa, e verificano che il meccanismo $P_e(R \mid s, a)$ sia stabile
al variare dell'ambiente $e \in \mathcal{E}$.

Questa analisi e' sufficiente quando le dinamiche di transizione
$P_e(s' \mid s, a)$ sono anch'esse invarianti, cioe' quando ambienti diversi
differiscono solo nella funzione di ricompensa o nel rumore esogeno.  Tuttavia,
in molti problemi fisici realistici le dinamiche di transizione **variano
significativamente** con i parametri ambientali: massa, lunghezza, gravita',
attrito, ecc.

Il problema concreto e' il seguente: il framework originale, nella Fase 3,
fornisce i parametri ambientali $(g, m, l, \ldots)$ come "contesto" additivo
alla policy -- concatenandoli alle feature di stato prima del trunk della rete
neurale.  Questo approccio ha tre limiti fondamentali:

1. **Incapacita' di estrapolazione**: la policy impara una funzione non
   strutturata del contesto su un numero finito di ambienti di training; non
   puo' generalizzare a combinazioni di parametri mai osservate.

2. **Mancanza di conoscenza strutturale**: la relazione tra parametri fisici ed
   efficacia delle azioni e' tipicamente **moltiplicativa** (es. nel pendolo,
   l'accelerazione angolare scala come $\tau / (ml^2)$), ma il conditioning
   additivo non codifica questo prior induttivo.

3. **Violazione dell'invarianza astratta**: concatenando il contesto al trunk,
   la policy astratta $\pi_{\text{abs}}$ diventa funzione dell'ambiente,
   vanificando il principio di invarianza che e' il fondamento del framework.

## 2. Lacuna nel Framework Originale

Il framework originale definisce la famiglia di ambienti (Definizione 2.1)
come un insieme di MDP che condividono spazio degli stati, spazio delle azioni
e **grafo causale**, ma possono differire nelle dinamiche $P_e$ e nella
ricompensa $R_e$.  L'analisi di invarianza (Sezione 3.3, IRM) e la selezione
delle feature (Sezione 3.6, Fase 2) testano esplicitamente solo l'invarianza
del meccanismo di ricompensa.

Non esiste, nel framework originale, alcun test di invarianza per i
**meccanismi di transizione**, ne' alcun meccanismo per sfruttare la struttura
della variabilita' delle dinamiche ai fini dell'ottimizzazione della policy.

La Sezione 3.6, Fase 3 prescrive:

> Sample environments [...] Collect trajectories [...] Compute gradients w.r.t.
> worst-case return, variance penalty, complexity penalty, constraint violations

ma non specifica come il conditioning ambientale debba interagire con la policy
quando le dinamiche variano.  Il condizionamento additivo (concatenazione del
contesto) e' un'assunzione implementativa non giustificata teoricamente.

## 3. Estensione: Sezione 3.7

L'estensione aggiunge una nuova sezione (3.7) con quattro definizioni, un
teorema, e due modifiche algoritmiche.

### 3.1. Invarianza del Meccanismo di Transizione (Definizione 3.7.1)

La feature di stato $s_i$ ha un **meccanismo di transizione invariante** se:

$$P_e(s_i' \mid s, a) = P_{e'}(s_i' \mid s, a) \quad \forall\, e, e' \in \mathcal{E}$$

Questo e' il duale della condizione di invarianza della ricompensa gia' presente
nel framework.  Si testa empiricamente con lo stesso approccio LOEO $R^2$ usato
nella Fase 2 per la ricompensa, ma con target di predizione $s_i'$ anziche' $R$.

**Razionale**: distinguere le dimensioni dello stato con dinamiche invarianti
(che non necessitano di normalizzazione) da quelle con dinamiche varianti
(che beneficiano della normalizzazione dell'azione).

### 3.2. Scala delle Dinamiche (Definizione 3.7.2)

La **scala delle dinamiche** dell'ambiente $e$ e':

$$D_e = \| B_e \|_F$$

dove $B_e \in \mathbb{R}^{d_s \times d_a}$ e' la matrice dei coefficienti
d'azione ottenuta dall'approssimazione lineare:

$$\Delta s_i \approx b_0 + W_s \cdot s + B_e[i, :] \cdot a$$

e $\|\cdot\|_F$ denota la norma di Frobenius.

**Razionale**: $D_e$ cattura "quanto una unita' di azione influenza le
transizioni di stato" nell'ambiente $e$.  E' una misura scalare
dell'efficacia dell'azione, stimabile in modo robusto con regressione lineare
sui dati di esplorazione gia' raccolti nella Fase 1.  L'uso della norma di
Frobenius (anziche' operatore o traccia) bilancia tutti gli effetti
dell'azione su tutte le dimensioni dello stato.

La **scala di riferimento** e' la media sugli ambienti di training:

$$D_{\text{ref}} = \mathbb{E}_e[D_e] = \frac{1}{|\mathcal{E}|} \sum_{e} D_e$$

### 3.3. Invarianza Astratta della Policy (Definizione 3.7.3)

Una policy $\pi$ ha **invarianza astratta** se puo' essere decomposta come:

$$\pi(a \mid s; e) = \mathcal{N}_e\bigl(\pi_{\text{abs}}(a \mid s)\bigr)$$

dove $\pi_{\text{abs}}$ e' una **policy astratta invariante** (indipendente da
$e$) e $\mathcal{N}_e$ e' un **normalizzatore dipendente dalle dinamiche** che
riscala le azioni astratte in azioni fisiche.

Per policy Gaussiane con squashing $\tanh$, il normalizzatore agisce sulla
distribuzione pre-squash:

$$\mu_e = r_e \cdot \mu_{\text{abs}}, \qquad \sigma_e = r_e \cdot \sigma_{\text{abs}}$$

dove $r_e = D_e / D_{\text{ref}}$ e' il **rapporto delle dinamiche**.

**Razionale**: la decomposizione impone un **prior induttivo moltiplicativo** --
la policy astratta ragiona in uno spazio normalizzato in cui l'effetto di una
"unita' di azione astratta" e' lo stesso in tutti gli ambienti.  Il
normalizzatore trasforma questa azione astratta nell'azione fisica appropriata
per l'ambiente corrente.

Scalare sia $\mu$ che $\sigma$ con lo stesso rapporto $r_e$ e' necessario per
la coerenza: $\text{Normal}(\mu \cdot r, \sigma \cdot r)$ e' equivalente a
moltiplicare la variabile aleatoria per $r$.  L'incertezza della policy astratta
deve essere espressa nelle stesse unita' dell'effetto astratto.

### 3.4. Teorema di Invarianza (Teorema 3.7.4)

Se (1) il meccanismo di ricompensa $P_e(R \mid s, a)$ e' invariante in
$\mathcal{E}$, e (2) le dinamiche di transizione sono
**azione-moltiplicative**, cioe':

$$f_s(s, a; e) = g(s) + B_e \cdot a + U_s$$

allora la policy astratta ottima $\pi_{\text{abs}}^*$ che massimizza il
rendimento nel caso peggiore e' invariante: $\pi_{\text{abs}}^*(a \mid s)$
non dipende da $e$.

**Idea della dimostrazione**: sotto dinamiche azione-moltiplicative, applicare
il normalizzatore $\mathcal{N}_e$ all'azione astratta $\tilde{a}$ produce
l'azione fisica $a = r_e \cdot \tilde{a}$, che genera la transizione effettiva:

$$B_e \cdot a = B_e \cdot r_e \cdot \tilde{a} = B_e \cdot \frac{D_e}{D_{\text{ref}}} \cdot \tilde{a}$$

Poiche' $D_e = \|B_e\|_F$, il termine $B_e \cdot D_e / D_{\text{ref}}$
equalizza l'ampiezza degli effetti delle azioni nello spazio astratto (a meno
di effetti direzionali nella struttura di $B_e$).  Dato che la ricompensa
dipende dallo stato (meccanismo invariante), e le dinamiche effettive nello
spazio astratto sono equalizzate, la policy astratta ottima e' invariante.

**Limiti del teorema**: il risultato e' esatto solo per dinamiche linearmente
azione-moltiplicative.  Per sistemi non lineari (la maggior parte dei sistemi
fisici reali), la decomposizione e' un'approssimazione locale il cui errore
cresce con l'ampiezza delle azioni e la non-linearita' delle dinamiche.

## 4. Modifiche Algoritmiche

### 4.1. Fase 2.5: Analisi delle Dinamiche di Transizione

Una nuova fase viene inserita tra la Selezione delle Feature (Fase 2) e
l'Ottimizzazione della Policy (Fase 3):

1. **Test LOEO di Transizione**: per ogni dimensione $s_i$, calcola
   $R^2$ LOEO della predizione di $s_i'$ da $(s, a)$.  Le dimensioni con
   $R^2$ basso hanno dinamiche varianti.

2. **Stima della Scala delle Dinamiche**: per ogni ambiente $e$, stima
   $B_e$ tramite regressione lineare su $\Delta s \sim s + a$ e calcola
   $D_e = \|B_e\|_F$ e la scala di riferimento
   $D_{\text{ref}} = \text{mean}(D_e)$.

**Razionale**: questa fase riusa i dati di esplorazione gia' raccolti nella
Fase 1 (non richiede interazioni aggiuntive con l'ambiente) e produce le
quantita' necessarie per la normalizzazione nella Fase 3.  Il costo
computazionale e' trascurabile rispetto alla scoperta causale e all'addestramento
della policy.

### 4.2. Fase 3 Modificata: Ottimizzazione con Normalizzazione delle Dinamiche

Quando la normalizzazione delle dinamiche e' attiva:

- Il **trunk della policy** riceve solo le feature di stato (nessun contesto),
  rendendo la policy astratta $\pi_{\text{abs}}$ invariante **per costruzione**.

- Il **contesto** (parametri ambientali) fluisce esclusivamente attraverso un
  **predittore delle dinamiche** appreso, che predice $\hat{D}_e$ e da cui si
  calcola il rapporto $r_e = \hat{D}_e / D_{\text{ref}}$.

- Una **loss ausiliaria di predizione delle dinamiche**
  $L_{\text{dyn}} = \| \hat{D}_e - D_e \|^2$ viene aggiunta per facilitare
  l'apprendimento del predittore.

**Razionale dell'esclusivita' del contesto**: se il contesto venisse fornito
sia al trunk che al predittore delle dinamiche, la rete potrebbe imparare un
"shortcut" additivo che bypassa la normalizzazione moltiplicativa, vanificando
il prior induttivo.  L'esclusivita' forza la separazione tra la policy astratta
(invariante, basata solo sullo stato) e il normalizzatore (dipendente
dall'ambiente, basato solo sul contesto).

**Razionale della loss ausiliaria**: l'apprendimento end-to-end tramite PPO
del predittore delle dinamiche e' possibile ma lento, poiche' il gradiente
della policy rispetto al predittore passa attraverso il rapporto $r_e$ e la
distribuzione delle azioni.  La loss ausiliaria $L_{\text{dyn}}$ fornisce un
segnale diretto di supervisione (le scale $D_e$ sono note dai dati di
esplorazione), accelerando significativamente la convergenza del predittore.

## 5. Relazione con il Framework Originale

L'estensione e' **strettamente additiva**: non modifica alcuna definizione o
risultato preesistente, ma aggiunge una nuova dimensione di analisi che era
assente.

| Aspetto | Framework originale (3.1--3.6) | Estensione (3.7) |
|---------|-------------------------------|-------------------|
| Oggetto dell'invarianza | Meccanismo di ricompensa $P_e(R \mid s,a)$ | Meccanismo di transizione $P_e(s' \mid s,a)$ |
| Test di invarianza | LOEO $R^2$ su ricompensa | LOEO $R^2$ su stato successivo |
| Conditioning ambientale | Additivo (concatenazione del contesto al trunk) | Moltiplicativo (scaling dell'azione via predittore) |
| Prior induttivo sulla policy | La policy invariante usa le stesse feature in tutti gli ambienti | La policy astratta opera in uno spazio normalizzato dove l'efficacia dell'azione e' uniforme |
| Ruolo dei parametri ambientali | Contesto generico | Input del predittore delle dinamiche |

La scelta di attivare o meno la normalizzazione e' controllata da un flag.
Per ambienti con dinamiche invarianti (tutte le dimensioni $s_i$ con alto LOEO
$R^2$ di transizione), la normalizzazione non e' necessaria e il framework
si comporta esattamente come la versione originale.

## 6. Limitazioni e Ipotesi

1. **Dinamiche azione-moltiplicative**: il Teorema 3.7.4 assume che la
   dipendenza dall'ambiente nelle dinamiche sia lineare nell'azione.  Questa
   e' una buona approssimazione per molti sistemi fisici (pendolo, carrello,
   braccio robotico), ma non vale in generale.

2. **Stimabilita' di $B_e$**: la stima tramite regressione lineare e' accurata
   solo se la relazione $\Delta s \sim a$ e' approssimativamente lineare nei
   dati di esplorazione.  Per sistemi fortemente non lineari, metodi di stima
   piu' sofisticati potrebbero essere necessari.

3. **Scalare vs. tensoriale**: l'uso di un singolo scalare $D_e$ come scala
   delle dinamiche comprime l'informazione direzionale di $B_e$.  Per sistemi
   con spazi d'azione multi-dimensionali dove le diverse componenti dell'azione
   scalano in modo diverso, una normalizzazione per componente potrebbe essere
   piu' appropriata.

4. **Generalizzazione out-of-distribution**: il predittore delle dinamiche
   $\hat{D}_e$ e' una rete neurale addestrata su un numero finito di ambienti.
   La sua capacita' di estrapolazione a parametri ambientali lontani dalla
   distribuzione di training dipende dalla regolarita' della funzione
   $\theta \mapsto D(\theta)$ e dalla copertura dei dati di training.
