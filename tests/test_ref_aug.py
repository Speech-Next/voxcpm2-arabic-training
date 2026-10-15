"""teacher-v3 (#5): corpus-weighted sampling and reference augmentation (CPU)."""

import math
import random
from collections import Counter

import pytest
import torch

from arabic_voxcpm.train import augment as A
from arabic_voxcpm.train import dataset as D
from arabic_voxcpm.train import pack

# --- corpus weights -------------------------------------------------------------------------------------------------


def test_parse_weights():
    w = D.parse_weights("synth=2, sada=0.5,synth:hejazi=3,sada:khaliji=0.25")
    assert w == {"synth": 2.0, "sada": 0.5, "synth:hejazi": 3.0, "sada:gulf": 0.25}  # aliases resolve
    assert D.parse_weights(None) == {} and D.parse_weights("") == {}
    for bad in ("voxtube=2", "synth:klingon=1", "sada=-1"):
        with pytest.raises(ValueError):
            D.parse_weights(bad)
    rows = [{"source": "synth", "dialect": "hejazi"}, {"source": "synth", "dialect": "msa"},
            {"source": "lahgtna", "dialect": "egyptian"}, {"source": "sada", "dialect": "gulf"}]
    assert D.row_weights(rows, w) == [3.0, 2.0, 1.0, 0.25]


class FakeSet:
    def __init__(self, lengths):
        self._len = lengths

    def __len__(self):
        return len(self._len)

    def length(self, i):
        return self._len[i]


def _rows(n_per=(300, 200, 100)):
    rows = []
    for c, n in zip(("lahgtna", "sada", "synth"), n_per):
        rows += [{"source": c, "dialect": "najdi" if c != "lahgtna" else "egyptian", "frames": 100 + 7 * i % 150}
                 for i in range(n)]
    return rows


def test_unweighted_sampler_unchanged():
    rng = random.Random(0)
    ds = FakeSet([rng.randint(20, 400) for _ in range(700)])
    s = D.TokenBudgetBatches(ds, 2000, seed=3)
    flat = sorted(i for b in s.batches() for i in b)
    assert flat == list(range(len(ds)))  # every row once


def test_weighted_sampler_deterministic_and_proportional():
    rows = _rows()
    ds = FakeSet([60 + r["frames"] // 4 for r in rows])
    w = D.row_weights(rows, D.parse_weights("synth=2.5,sada=0.5"))
    a = D.TokenBudgetBatches(ds, 3000, seed=1, weights=w)
    b = D.TokenBudgetBatches(ds, 3000, seed=1, weights=w)
    assert a.batches() == b.batches()  # same seed and epoch: same batches
    b.epoch = 1
    assert a.batches() != b.batches()  # another epoch: other draws
    counts = Counter()
    epochs = 40
    for e in range(epochs):
        a.epoch = e
        bs = a.batches()
        for batch in bs:
            assert max(ds.length(i % len(ds)) for i in batch) * len(batch) <= 3000 or len(batch) == 1
            for i in batch:
                counts[rows[i % len(ds)]["source"]] += 1
        idx = [i for batch in bs for i in batch]
        assert len(idx) == len(set(idx))  # copies carry distinct indices
        # a weight >= 1 row is always drawn at least floor(w) times
        per_row = Counter(i % len(ds) for i in idx)
        assert all(per_row[i] >= 2 for i, r in enumerate(rows) if r["source"] == "synth")
        assert all(per_row[i] == 1 for i, r in enumerate(rows) if r["source"] == "lahgtna")
    for c, n, wc in (("lahgtna", 300, 1.0), ("sada", 200, 0.5), ("synth", 100, 2.5)):
        assert counts[c] / epochs == pytest.approx(n * wc, rel=0.05), c


def test_effective_hours():
    rows = [{"source": "synth", "dialect": "najdi", "frames": 25 * 3600}, {"source": "sada", "dialect": "najdi",
                                                                            "frames": 25 * 1800}]
    assert D.hours(rows) == {"sada": 0.5, "sada:najdi": 0.5, "synth": 1.0, "synth:najdi": 1.0}
    assert D.hours(rows, [2.0, 0.5]) == {"sada": 0.25, "sada:najdi": 0.25, "synth": 2.0, "synth:najdi": 2.0}


# --- the dataset's reference marks ----------------------------------------------------------------------------------


class Store(dict):
    pass


def _trainset(p_ref_aug, variants=None, n=12):
    g = torch.Generator().manual_seed(0)
    rows, store = [], Store()
    for i in range(n):
        uid = f"u{i}"
        frames = 60 + 13 * i
        store[uid] = torch.randn(frames, 64, generator=g).half()
        rows.append({"uid": uid, "source": "synth", "dialect": "najdi", "speaker": f"s{i % 3}", "text": "هلا والله",
                     "frames": frames})
    return D.TrainSet(rows, store, lambda t: [ord(c) % 97 for c in t], p_ref=1.0, p_no_tag=0.0, seed=5,
                      p_ref_aug=p_ref_aug, aug_variants=variants)


def test_ref_aug_marks():
    clean, marked = _trainset(0.0), _trainset(1.0)
    for i in range(len(clean)):
        a, b = clean[i], marked[i]
        assert "ref_aug" not in a  # off: the samples as before
        torch.testing.assert_close(a["audio_feats"], b["audio_feats"])  # marking draws after the window
        ref = (b["audio_mask"] == 1) & (b["loss_mask"] == 0)
        m = b["ref_aug"]
        assert bool((m > 0).eq(ref).all())  # exactly the reference patches
        n = int(m.sum())
        pos = ref.nonzero()[:, 0]
        frames = b["audio_feats"][pos].reshape(-1, 64)
        assert bool((frames[n:] == 0).all()) and bool((frames[n - 1] != 0).any())  # real frames, then the padding
    half = _trainset(0.5)
    hit = sum(int(half[i]["ref_aug"].any()) for i in range(len(half)))
    assert 0 < hit < len(half)


def test_weighted_copies_draw_their_own_reference():
    ds = _trainset(0.0)
    n = len(ds)
    assert any(not torch.equal(ds[i]["audio_feats"], ds[i + n]["audio_feats"]) for i in range(n))
    torch.testing.assert_close(ds[3]["audio_feats"], ds[3]["audio_feats"])  # deterministic


def test_precomputed_variants():
    base = _trainset(0.0)
    variant = Store({u: torch.full_like(z, 7) for u, z in base.store.items()})
    ds = _trainset(1.0, variants=[variant])
    for i in range(len(ds)):
        s = ds[i]
        assert "ref_aug" not in s
        ref = (s["audio_mask"] == 1) & (s["loss_mask"] == 0)
        frames = s["audio_feats"][ref].reshape(-1, 64)
        assert bool((frames[frames.abs().sum(1) > 0] == 7).all())  # the reference comes from the variant
        tgt = s["audio_feats"][s["loss_mask"] == 1]
        assert not bool((tgt == 7).any())  # the target never


# --- degradations ---------------------------------------------------------------------------------------------------


def _speechlike(B=3, n=32000, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / A.SR
    f0 = torch.tensor([140.0, 210.0, 330.0])[:B, None]
    env = (0.5 + 0.5 * torch.sin(2 * math.pi * 3 * t))[None]
    x = sum(torch.sin(2 * math.pi * k * f0 * t[None]) / k for k in range(1, 6)) * env
    return 0.2 * x + 0.01 * torch.randn(B, n, generator=g)


def _params(B, **kw):
    cfg = A.AugConfig(**kw)
    return A.draw(random.Random(0), B, cfg)


def test_degrade_shapes_lengths_and_level():
    x = _speechlike()
    lengths = torch.tensor([32000, 20000, 9000])
    p = _params(3, p_noise=1, p_reverb=1, p_lowpass=1)
    y = A.degrade(x, lengths, p)
    assert y.shape == x.shape and torch.isfinite(y).all()
    for i, n in enumerate(lengths.tolist()):
        assert bool((y[i, n:] == 0).all())
    torch.testing.assert_close(A.rms(y, lengths), A.rms(x * (torch.arange(32000)[None] < lengths[:, None]), lengths),
                               rtol=1e-3, atol=1e-6)
    assert float(y.abs().max()) <= 0.99 + 1e-6
    torch.testing.assert_close(A.degrade(x, lengths, p), y)  # deterministic
    p2 = dict(p, seed=[s + 1 for s in p["seed"]])
    assert not torch.allclose(A.degrade(x, lengths, p2), y)


@pytest.mark.parametrize("snr", [5.0, 15.0, 25.0])
def test_noise_snr(snr):
    x = _speechlike()
    lengths = torch.full((3,), x.shape[1])
    p = _params(3, p_noise=1, p_reverb=0, p_lowpass=0, snr_db=(snr, snr))
    y = A.degrade(x, lengths, p)
    c = (y * x).sum(1) / (x * x).sum(1)  # y = c (x + n), x and n nearly orthogonal
    measured = 10 * torch.log10((c[:, None] * x).pow(2).sum(1) / (y - c[:, None] * x).pow(2).sum(1))
    assert measured.tolist() == pytest.approx([snr] * 3, abs=0.5)


def test_coloured_noise_slope():
    n = 1 << 16
    for beta in (0.0, 1.0, 2.0):
        w = A.coloured_noise(4, n, torch.full((4,), beta), [1, 2, 3, 4], "cpu")
        assert w.pow(2).mean(1).sqrt().tolist() == pytest.approx([1.0] * 4, rel=1e-4)
        psd = torch.fft.rfft(w, dim=1).abs().pow(2).mean(0)
        f = torch.fft.rfftfreq(n, 1 / A.SR)
        lo = psd[(f > 200) & (f < 400)].mean()
        hi = psd[(f > 3200) & (f < 6400)].mean()
        assert float(10 * torch.log10(lo / hi)) == pytest.approx(10 * beta * math.log10(16), abs=1.5)


@pytest.mark.parametrize("rt60", [0.2, 0.5, 0.8])
def test_rir_decay(rt60):
    h = A.rir(torch.full((2,), rt60), torch.full((2,), 6.0), [11, 12], "cpu")
    tail = h[:, 1:]
    edc = tail.pow(2).flip(1).cumsum(1).flip(1)  # Schroeder backward integral
    db = 10 * torch.log10(edc / edc[:, :1])
    for row in db:
        t5 = int((row > -5).sum())
        t25 = int((row > -25).sum())
        assert 3 * (t25 - t5) / A.SR == pytest.approx(rt60, rel=0.15)  # T20 x 3
    drr = 10 * torch.log10(h[:, 0] ** 2 / tail.pow(2).sum(1))
    assert drr.tolist() == pytest.approx([6.0, 6.0], abs=1e-3)


def test_lowpass_and_resample():
    n = 1 << 15
    w = torch.randn(2, n, generator=torch.Generator().manual_seed(0))
    y = A.lowpass(w, torch.tensor([3000.0, 6000.0]))
    psd = torch.fft.rfft(y, dim=1).abs().pow(2)
    f = torch.fft.rfftfreq(n, 1 / A.SR)
    for i, fc in enumerate((3000, 6000)):
        passband = psd[i, (f > 200) & (f < fc - 400)].mean()
        stop = psd[i, f > fc + 400].mean()
        assert float(10 * torch.log10(passband / stop)) > 40
    t = torch.arange(48000 * 2) / 48000
    s = torch.sin(2 * math.pi * 1000 * t) + torch.sin(2 * math.pi * 12000 * t)  # 12 kHz: above the 16 kHz Nyquist
    d = A.to_16k(s[None])[0]
    assert d.shape[0] == 32000
    ref = torch.sin(2 * math.pi * 1000 * torch.arange(32000) / 16000)
    assert float((d[1000:-1000] - ref[1000:-1000]).abs().max()) < 0.02  # 1 kHz kept at its level, 12 kHz gone


def test_chunks():
    lengths = [250, 80, 120, 250, 75, 300, 90]
    cs = A.chunks(lengths, 400)
    assert sorted(i for c in cs for i in c) == list(range(len(lengths)))
    assert all(len(c) == 1 or max(lengths[i] for i in c) * len(c) <= 400 for c in cs)


# --- the step's augmenter on a packed batch (a stand-in AudioVAE) ---------------------------------------------------


class FakeVAE:
    """decode: 64 x F latents -> 1920 F samples (each frame's mean, held); encode: 640 samples -> 64 dims (the mean)."""

    def decode(self, z, sr=None):
        return z.mean(1, keepdim=True).repeat_interleave(1920, dim=2).tanh()

    def encode(self, x, sr):
        B, _, N = x.shape
        return x.reshape(B, 1, N // 640, 640).mean(3).expand(B, 64, N // 640).clone()


def test_augmenter_changes_only_marked_references():
    ds = _trainset(0.5)
    seqs = [ds[i] for i in range(len(ds))]
    batch = pack.collate(seqs)
    before = {k: v.clone() for k, v in batch.items()}
    aug = A.RefAugmenter(FakeVAE(), seed=0, chunk_frames=300)
    n = aug(batch, (3, 0))
    assert n == sum(int(s["ref_aug"].any()) for s in seqs) > 0
    marked = batch["ref_aug"] > 0
    feats, old = batch["audio_feats"], before["audio_feats"]
    assert torch.equal(feats[~marked], old[~marked])  # targets, unmarked references, text, padding: untouched
    assert not torch.equal(feats[marked], old[marked])
    for b in range(len(seqs)):
        m = batch["ref_aug"][b]
        if m.any():
            k = int(m.sum())
            frames = feats[b, m.nonzero()[:, 0]].reshape(-1, 64)
            assert bool((frames[k:] == 0).all())  # the window's padding frames stay zero
    again = {k: v.clone() for k, v in before.items()}
    aug(again, (3, 0))
    assert torch.equal(again["audio_feats"], feats)  # deterministic given the key
    other = {k: v.clone() for k, v in before.items()}
    aug(other, (4, 0))
    assert not torch.equal(other["audio_feats"], feats)


def test_sample_rep_is_weighted_copy():
    # teacher_data's sample(i, rep) and the weighted sampler's index i + rep * len draw the same sample
    ds = _trainset(0.0)
    ds.p_ref = 0.5  # some samples clone, some do not
    for i in range(len(ds)):
        for rep in range(3):
            seq, info = ds.sample(i, rep)
            other = ds[i + rep * len(ds)]
            assert seq.keys() == other.keys()
            for k in seq:
                torch.testing.assert_close(seq[k], other[k])
            assert (info["ref"] is None) == bool((seq["audio_mask"] == 1).eq(seq["loss_mask"] == 1).all())


def test_untagged_set_has_no_tags():
    ds = _trainset(0.0)
    plain = D.TrainSet(ds.rows, ds.store, lambda t: [ord(c) % 97 for c in t], p_ref=1.0, p_no_tag=0.0, seed=5,
                       tagged=False)
    assert plain.tagged_ids == plain.text_ids != ds.tagged_ids
    assert all(plain.sample(i)[1]["tagged"] for i in range(len(plain)))  # drawn as tagged, but the ids carry no tag
