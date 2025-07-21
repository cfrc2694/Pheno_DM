import gzip
import xml.etree.ElementTree as ET
from abc import ABC
from pathlib import Path

import numpy as np
from ROOT import TLorentzVector, TMath, TVector2, TVector3


class Particle(ABC):

    def __init__(self, **kwargs):

        self.tlv = TLorentzVector(0, 0, 0, 0)
        self.charge = kwargs.get("charge", 0.0)
        self.name = kwargs.get("name", "")
        self.kind = kwargs.get("kind", "")

        if not isinstance(self.tlv, TLorentzVector):
            raise TypeError("tlv must be a TLorentzVector object")
        if not isinstance(self.charge, float):
            raise TypeError("charge must be a float")
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not isinstance(self.kind, str):
            raise TypeError("kind must be a string")

    def get_charge(self):
        return self.charge

    def get_tlv(self):
        return self.tlv

    def get_name(self):
        return self.name

    def set_name(self, new_name: str) -> None:
        self.name = new_name

    @property
    def pt(self) -> float:

        tlv = self.tlv
        return tlv.Pt()

    @property
    def p(self) -> float:

        return self.tlv.P()

    @property
    def pl(self) -> float:

        p = self.tlv.P()
        pt = self.tlv.Pt()
        return TMath.Sqrt((p - pt) * (p + pt))

    @property
    def eta(self) -> float:

        tlv = self.tlv
        return tlv.Eta()

    @property
    def phi(self) -> float:

        phi = self.tlv.Phi()
        return phi

    @property
    def m(self) -> float:

        return self.tlv.M()

    @property
    def energy(self) -> float:

        return self.tlv.Energy()

    def set_good_tag(self, value):

        if value not in [0, 1]:
            raise ValueError("Error: good_tag value should be 0 or 1.")
        self.good_tag = value

    def get_good_tag(self, cuts):

        try:
            return self.good_tag
        except AttributeError:
            pass
        kin_cuts = cuts.get(self.kind)

        pt_min_cut = kin_cuts.get("pt_min_cut")
        pt_max_cut = kin_cuts.get("pt_max_cut")
        eta_min_cut = kin_cuts.get("eta_min_cut")
        eta_max_cut = kin_cuts.get("eta_max_cut")

        pt_cond = self.pt >= pt_min_cut
        if pt_max_cut:
            if not (pt_max_cut > pt_min_cut):
                raise Exception("Error: pt_max must be major than pt_min")
            pt_cond = pt_cond and (self.pt <= pt_max_cut)
        eta_cond = (self.eta >= eta_min_cut) and (self.eta <= eta_max_cut)

        if pt_cond and eta_cond:
            self.set_good_tag(1)
        else:
            self.set_good_tag(0)

        return self.good_tag

    # Delta methods
    def delta_R(self, v2):

        tlv1 = self.tlv
        tlv2 = v2.get_tlv()
        return tlv1.DeltaR(tlv2)

    def delta_eta(self, v2):

        tlv1 = self.tlv
        tlv2 = v2.get_tlv()
        return abs(tlv1.Eta() - tlv2.Eta())

    def delta_phi(self, v2):

        tlv1 = self.tlv
        tlv2 = v2.get_tlv()
        return abs(tlv1.DeltaPhi(tlv2))

    def delta_pt_scalar(self, v2):

        tlv1 = self.tlv
        tlv2 = v2.get_tlv()
        return tlv1.Pt() - tlv2.Pt()

    def delta_pt_vectorial(self, v2):

        tlv1 = self.tlv
        tlv2 = v2.get_tlv()
        a = TVector2(tlv1.Px(), tlv1.Py())
        b = TVector2(tlv2.Px(), tlv2.Py())
        c = a - b
        return c.Mod()

    def delta_p_vectorial(self, v2):

        tlv1 = self.tlv
        tlv2 = v2.get_tlv()
        a = TVector3(tlv1.Px(), tlv1.Py(), tlv1.Pz())
        b = TVector3(tlv2.Px(), tlv2.Py(), tlv2.Pz())
        c = a - b
        return c.Mag()


class LHEParticle(Particle):
    def __init__(self, pdgid, spin, px=0, py=0, pz=0, energy=0, mass=0):
        self.pdgid = pdgid
        typep = abs(pdgid)
        if (typep == 2) or (typep == 4) or (typep == 6):
            charge = np.sign(pdgid) * 2.0 / 3.0
        elif (typep == 1) or (typep == 3) or (typep == 5):
            charge = -np.sign(pdgid) * 1.0 / 3.0
        elif (typep == 11) or (typep == 13) or (typep == 15):
            charge = -np.sign(pdgid)
        else:
            charge = 0.0

        super().__init__(charge=float(charge))
        self.px = px
        self.py = py
        self.pz = pz
        # self.energy = energy
        self.tlv.SetPxPyPzE(px, py, pz, energy)
        self.mass = mass
        self.spin = spin


class Event:
    def __init__(self, num_particles):
        self.num_particles = num_particles
        self.particles = []

    def __addParticle__(self, particle):
        self.particles.append(particle)

    def getParticlesByIDs(self, idlist):
        partlist = [p for p in self.particles if p.pdgid in idlist]
        return partlist

    def getMissingET(self, idlist: list = [12, 14, 16, -12, -14, -16]):
        met_list = self.getParticlesByIDs(idlist)
        met_pdgid = 0
        met_spin = 0
        met_px = sum([p.px for p in met_list])
        met_py = sum([p.py for p in met_list])
        met_pz = 0  # Is Missing ET
        met_e = sum([p.e for p in met_list])
        met_mass = TLorentzVector(met_px, met_py, met_pz, met_e).M()
        met = LHEParticle(met_pdgid, met_spin, met_px, met_py, met_pz, met_e, met_mass)
        return met


class LHEFData:
    def __init__(self, version):
        self.version = version
        self.events = []

    def __addEvent__(self, event):
        self.events.append(event)

    def getParticlesByIDs(self, idlist):
        partlist = []
        for event in self.events:
            partlist.extend(event.getParticlesByIDs(idlist))
        return partlist


def readLHEF(path_to_file):
    with gzip.open(path_to_file, "rb") as lhe_file:
        tree = ET.parse(lhe_file)
        root = tree.getroot()
        childs = [child for child in root if child.tag == "event"]
    return childs


def get_event_by_child(child):
    lines = child.text.strip().split("\n")
    event_header = lines[0].strip()
    num_part = int(event_header.split()[0].strip())
    e = Event(num_part)
    for i in range(1, num_part + 1):
        part_data = lines[i].strip().split()
        if int(part_data[1]) != 1: # Only process final state particles
            continue

        p = LHEParticle(
            int(part_data[0]),  # pdg-id
            float(part_data[12]),  # spin
            float(part_data[6]),  # px
            float(part_data[7]),  # py
            float(part_data[8]),  # pz
            float(part_data[9]),  # E
            float(part_data[10]),  # m
        )
        e.__addParticle__(p)
    return e


def get_forest(glob: str = None, path_to_signal: str = None) -> list:

    if glob is None:
        raise ValueError("Please provide a glob to search the LHE files")
    if path_to_signal is None:
        raise ValueError("Please provide the path to the signal LHE files")

    def natural_sort(list):
        import re

        def convert(text):
            if text.isdigit():
                return int(text)
            else:
                return text.lower()

        def alphanum_key(key):
            return [convert(c) for c in re.split("([0-9]+)", key)]

        return sorted(list, key=alphanum_key)

    path_root = Path(path_to_signal)
    forest = [root_file.as_posix() for root_file in path_root.glob(glob)]
    return natural_sort(forest)
