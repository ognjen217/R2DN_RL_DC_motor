# Reference

## R2DN rad

Nicholas H. Barbara, Ruigang Wang i Ian R. Manchester,
“R2DN: Scalable Parameterization of Contracting and Lipschitz Recurrent Deep
Networks,” arXiv:2504.01250v2, 2026.

- Rad: <https://arxiv.org/abs/2504.01250>
- Zvanični kod: <https://github.com/nic-barbara/R2DN>
- Kod modela: <https://github.com/nic-barbara/R2DN/blob/main/robustnn/r2dn.py>
- Verzija proverena u Fazi 0:
  `5e65ac9bb5a057d41232162133ac1454865b965b`
- Licenca upstream koda: MIT

Rad definiše R2DN kao diskretni nelinearni state-space model koji povezuje LTI
sistem i 1-Lipschitz feedforward mrežu. Zvanični kod implementira
`ContractingR2DN` u JAX/Flax-u i izlaže jednokoračni poziv, inicijalizaciju
rekurentnog stanja i `simulate_sequence` rollout.

Upstream kod nije kopiran u ovom projektu u Fazi 0. Kasnija integracija mora
ostati vezana za konkretan commit. Ako se kod vendorizuje ili modifikuje,
originalni MIT copyright i dozvola moraju biti sačuvani.

