import pandas as pd
import pytest
from lxml import etree

from real_estate.ingest.rcn_build import _share
from real_estate.ingest.rcn_parse import _parse_feature, _ring_centroid

NS = (
    'xmlns:rcn="urn:gugik:specyfikacje:gmlas:rejestrcennieruchomosci:1.0" '
    'xmlns:gml="http://www.opengis.net/gml/3.2" xmlns:xlink="http://www.w3.org/1999/xlink"'
)


def test_ring_centroid_square():
    # square 0..10 given as (northing, easting) pairs
    e, n, area = _ring_centroid([0, 0, 0, 10, 10, 10, 10, 0, 0, 0])
    assert (e, n, area) == pytest.approx((5, 5, 100))


def test_share_parsing():
    s = _share(pd.Series(["1/1", "1/2", " 2 / 3 ", "bad", None]))
    assert s.iloc[0] == 1 and s.iloc[1] == 0.5
    assert s.iloc[2] == pytest.approx(2 / 3)
    assert s.iloc[3:].isna().all()


def test_parse_lokal_feature():
    xml = f"""<rcn:RCN_Lokal {NS} gml:id="L1">
      <rcn:idLokalu>146516_8.0620.149_BUD.1_LOK</rcn:idLokalu>
      <rcn:georeferencja><gml:Point gml:id="p" srsName="urn:ogc:def:crs:EPSG::2178">
        <gml:pos>5786000.0 7500000.0</gml:pos></gml:Point></rcn:georeferencja>
      <rcn:funkcjaLokalu>1</rcn:funkcjaLokalu>
      <rcn:powUzytkowaLokalu uom="m2">50.5</rcn:powUzytkowaLokalu>
      <rcn:adresBudynkuZLokalem xlink:href="A1"/>
    </rcn:RCN_Lokal>"""
    rec = _parse_feature(etree.fromstring(xml))
    assert rec["gml_id"] == "L1"
    assert rec["funkcjaLokalu"] == "1" and rec["powUzytkowaLokalu"] == "50.5"
    assert rec["adresBudynkuZLokalem"] == ["A1"]
    # 7500000 E / 5786000 N in PL-2000 zone 7 lies in central Warsaw
    assert 630_000 < rec["x_2180"] < 645_000 and 480_000 < rec["y_2180"] < 495_000
