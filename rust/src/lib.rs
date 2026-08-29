//! Noyau natif de pyardl : la récursion du DGP nul du bootstrap.
//!
//! CE QUI EST ICI, ET POURQUOI SEULEMENT CELA
//! ------------------------------------------
//! Ce crate ne réimplémente pas le bootstrap. Il porte **une seule
//! fonction**, choisie par la mesure et non par intuition : la
//! récursion qui régénère les trajectoires sous l'hypothèse nulle.
//!
//! Profilage de `bootstrap_bounds_test` à T = 1000, B = 9999, k = 3
//! (26,8 s sous cProfile) :
//!
//! | poste                        | temps  | part |
//! |------------------------------|--------|------|
//! | `numpy.linalg.qr`            |  9,5 s |  36% |
//! | **`simulate_paths`**         |  7,3 s |  27% |
//! | `numpy.stack` (design)       |  2,9 s |  11% |
//! | allocations dans la boucle   |  ~2 s  |   7% |
//!
//! Le QR est déjà du LAPACK : le réécrire ici appellerait le même
//! LAPACK, pour rien. `simulate_paths` est le seul poste où Python
//! coûte vraiment quelque chose, parce que la récursion est
//! **séquentielle en t** : NumPy ne peut pas la vectoriser sur le
//! temps, donc la boucle paie 1050 tours d'interpréteur et
//! d'allocation sur des tableaux (B, k).
//!
//! En Rust la même arithmétique devient une boucle sur `t` à
//! l'intérieur d'une boucle sur les réplications — l'ordre inverse. Le
//! travail d'une réplication tient alors dans le cache L2 (33 Ko à ces
//! dimensions), et les réplications sont indépendantes, donc
//! parallélisables.
//!
//! **Cela plafonne le gain de bout en bout à environ 1,4x** (loi
//! d'Amdahl sur 27 %). Ce n'est pas un détail à taire : le vrai coût
//! reste le QR, et l'optimisation suivante est algorithmique, pas de
//! langage.
//!
//! CE QUI N'EST PAS ICI
//! --------------------
//! Le tirage aléatoire. Les innovations sont **passées en argument**,
//! déjà tirées côté Python par un `numpy.random.Generator` à graine
//! explicite. Deux raisons : la reproductibilité reste sous le contrôle
//! d'un seul générateur, et surtout l'équivalence entre les deux
//! backends devient **exacte** — mêmes innovations, mêmes trajectoires
//! au bit près — au lieu d'être seulement distributionnelle. Un test de
//! Kolmogorov-Smirnov aurait laissé passer un écart systématique de
//! 1e-9 ; l'égalité à 1e-12 ne le laisse pas passer.
//!
//! Le rappel `expand` de la décomposition NARDL non plus : c'est un
//! objet Python appelé à chaque période, et le faire traverser la
//! frontière 1050 fois coûterait plus que ce que la boucle économise.
//! Le chemin NumPy le traite, et le dispatch y retombe.

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Ce que la récursion rend : les niveaux de `y` et ceux des
/// régresseurs, dans les dispositions que NumPy attend côté Python.
type Paths<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray3<f64>>);

/// Coefficients du DGP nul, aplatis pour traverser la frontière.
struct Dgp<'a> {
    x_const: &'a [f64],
    x_ar: &'a [f64], // (r, k, k) en ordre C
    y_const: f64,
    y_trend: f64,
    psi: &'a [f64],
    omega_flat: &'a [f64],
    omega_offsets: &'a [i64],
    q: &'a [i64],
    first_lag: usize,
    p: usize,
    r: usize,
    k: usize,
    k_cond: usize,
    det: Option<&'a [f64]>,
    burn_in: usize,
    n_total: usize,
    lag_max: usize,
}

/// Une réplication : la récursion, puis les sommes cumulées.
///
/// `dx` et `dy` sont des tampons locaux réutilisés par le thread ; à
/// B = 9999 cela évite 20 000 allocations dont NumPy, lui, ne peut pas
/// se passer puisqu'il produit un tableau par période.
#[allow(clippy::too_many_arguments)]
fn one_replication(
    dgp: &Dgp,
    inn: &[f64], // (n_total, 1 + k) de cette réplication, ordre C
    y0: f64,
    x0: &[f64],
    dx: &mut [f64], // (n_total, k)
    dy: &mut [f64], // (n_total,)
    y_out: &mut [f64],
    x_out: &mut [f64],
) {
    let k = dgp.k;
    let n_eq = 1 + k;
    dx[..dgp.lag_max * k].fill(0.0);
    dy[..dgp.lag_max].fill(0.0);

    for t in dgp.lag_max..dgp.n_total {
        // dx[t] = x_const + sum_i dx[t-i-1] @ x_ar[i]' + inn[t, 1:]
        for c in 0..k {
            let mut acc = dgp.x_const[c];
            for i in 0..dgp.r {
                let lagged = &dx[(t - i - 1) * k..(t - i) * k];
                let block = &dgp.x_ar[i * k * k + c * k..i * k * k + (c + 1) * k];
                for d in 0..k {
                    acc += lagged[d] * block[d];
                }
            }
            dx[t * k + c] = acc + inn[t * n_eq + 1 + c];
        }

        let mut val = dgp.y_const;
        if dgp.y_trend != 0.0 {
            val += dgp.y_trend * (t as f64 - dgp.burn_in as f64 + 1.0);
        }
        if let Some(det) = dgp.det {
            val += det[t];
        }
        for i in 1..dgp.p {
            val += dgp.psi[i - 1] * dy[t - i];
        }
        for j in 0..dgp.k_cond {
            let start = dgp.omega_offsets[j] as usize;
            let q_j = dgp.q[j] as usize;
            for i in dgp.first_lag..q_j {
                val += dgp.omega_flat[start + i - dgp.first_lag] * dx[(t - i) * k + j];
            }
        }
        dy[t] = val + inn[t * n_eq];
    }

    // Niveaux : somme cumulée après le rodage, décalée des valeurs
    // initiales observées.
    let mut acc_y = y0;
    for (out_t, t) in (dgp.burn_in..dgp.n_total).enumerate() {
        acc_y += dy[t];
        y_out[out_t] = acc_y;
    }
    let mut acc_x = vec![0.0f64; k];
    acc_x.copy_from_slice(x0);
    for (out_t, t) in (dgp.burn_in..dgp.n_total).enumerate() {
        for c in 0..k {
            acc_x[c] += dx[t * k + c];
            x_out[out_t * k + c] = acc_x[c];
        }
    }
}

/// Régénère `B` trajectoires sous le DGP nul.
///
/// Chaque argument est déjà validé côté Python : cette fonction est un
/// noyau, pas une API publique. Les seules vérifications faites ici
/// portent sur ce qui provoquerait un accès hors bornes.
#[pyfunction]
#[pyo3(signature = (
    inn, x_const, x_ar, y_const, y_trend, psi, omega_flat, omega_offsets,
    q, first_lag, p, det_contrib, burn_in, y0, x0
))]
#[allow(clippy::too_many_arguments)]
fn simulate_paths<'py>(
    py: Python<'py>,
    inn: PyReadonlyArray3<'py, f64>,
    x_const: PyReadonlyArray1<'py, f64>,
    x_ar: PyReadonlyArray3<'py, f64>,
    y_const: f64,
    y_trend: f64,
    psi: PyReadonlyArray1<'py, f64>,
    omega_flat: PyReadonlyArray1<'py, f64>,
    omega_offsets: PyReadonlyArray1<'py, i64>,
    q: PyReadonlyArray1<'py, i64>,
    first_lag: usize,
    p: usize,
    det_contrib: Option<PyReadonlyArray1<'py, f64>>,
    burn_in: usize,
    y0: f64,
    x0: PyReadonlyArray1<'py, f64>,
) -> PyResult<Paths<'py>> {
    let inn_arr = inn.as_array();
    let (n_rep, n_total, n_eq) = inn_arr.dim();
    let k = x_const.len()?;
    if n_eq != 1 + k {
        return Err(PyValueError::new_err(format!(
            "innovations has {n_eq} columns for {k} regressors; expected {}.",
            1 + k
        )));
    }
    if burn_in >= n_total {
        return Err(PyValueError::new_err(
            "burn_in leaves no observation out of the simulated periods.",
        ));
    }
    let x_ar_arr = x_ar.as_array();
    let r = x_ar_arr.dim().0;
    let q_slice = q.as_slice()?;
    let k_cond = q_slice.len();
    let lag_max = [p, q_slice.iter().copied().max().unwrap_or(0) as usize, r, 1]
        .into_iter()
        .max()
        .unwrap();
    if lag_max > n_total {
        return Err(PyValueError::new_err(
            "the lag order exceeds the number of simulated periods.",
        ));
    }
    if k_cond > k {
        return Err(PyValueError::new_err(
            "the conditional block has more columns than the marginal one; \
             this path requires expand=None.",
        ));
    }

    let inn_owned = inn_arr.to_owned();
    let inn_std = inn_owned
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset()
        .0;
    let x_ar_owned = x_ar_arr.to_owned();
    let x_ar_std = x_ar_owned
        .as_standard_layout()
        .into_owned()
        .into_raw_vec_and_offset()
        .0;
    let det_owned: Option<Vec<f64>> = match &det_contrib {
        Some(d) => {
            let v = d.as_slice()?.to_vec();
            if v.len() != n_total {
                return Err(PyValueError::new_err(
                    "det_contrib must have one value per simulated period.",
                ));
            }
            Some(v)
        }
        None => None,
    };

    let dgp = Dgp {
        x_const: x_const.as_slice()?,
        x_ar: &x_ar_std,
        y_const,
        y_trend,
        psi: psi.as_slice()?,
        omega_flat: omega_flat.as_slice()?,
        omega_offsets: omega_offsets.as_slice()?,
        q: q_slice,
        first_lag,
        p,
        r,
        k,
        k_cond,
        det: det_owned.as_deref(),
        burn_in,
        n_total,
        lag_max,
    };
    let x0_slice = x0.as_slice()?;
    let n_keep = n_total - burn_in;

    let mut y_star = vec![0.0f64; n_rep * n_keep];
    let mut x_star = vec![0.0f64; n_rep * n_keep * k];

    // Les réplications sont indépendantes : la dépendance temporelle est
    // interne à chacune. C'est ce qui rend cette boucle parallèle sans
    // aucune synchronisation.
    py.allow_threads(|| {
        y_star
            .par_chunks_mut(n_keep)
            .zip(x_star.par_chunks_mut(n_keep * k))
            .enumerate()
            .for_each(|(b, (y_out, x_out))| {
                let mut dx = vec![0.0f64; n_total * k];
                let mut dy = vec![0.0f64; n_total];
                let inn_b = &inn_std[b * n_total * n_eq..(b + 1) * n_total * n_eq];
                one_replication(&dgp, inn_b, y0, x0_slice, &mut dx, &mut dy, y_out, x_out);
            });
    });

    let y_arr = Array2::from_shape_vec((n_rep, n_keep), y_star)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let x_arr = Array3::from_shape_vec((n_rep, n_keep, k), x_star)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((y_arr.into_pyarray(py), x_arr.into_pyarray(py)))
}

/// Nombre de threads que rayon utilisera — rapporté pour que la
/// documentation puisse citer un chiffre mesuré plutôt qu'annoncé.
#[pyfunction]
fn thread_count() -> usize {
    rayon::current_num_threads()
}

/// Le nom de cette fonction EST le nom du module importable : Python
/// cherche `PyInit__rust` dans `pyardl/_rust.pyd`. Le renommer sans
/// renommer le fichier produit un `ImportError` dont le message ne
/// mentionne ni l'un ni l'autre.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Native kernel for pyardl's bootstrap null DGP.")?;
    m.add_function(wrap_pyfunction!(simulate_paths, m)?)?;
    m.add_function(wrap_pyfunction!(thread_count, m)?)?;
    Ok(())
}
