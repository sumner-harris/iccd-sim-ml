from __future__ import annotations

from iccd_sim_ml.atomic.nist import parse_ionization_csv, parse_levels_html


def test_parse_nist_level_html_and_bound_filter() -> None:
    html = """
    <html><body><p>NIST ASD (ver.&nbsp;5.12)</p><p>3 Levels Found</p>
    <table><tbody>
    <tr class="bsl"><td>3d<sup>10</sup>4s</td><td><sup>2</sup>S</td>
      <td><sup>1</sup>/<sub>2</sub></td><td>2</td><td>0.000000</td><td>0</td>
      <td></td><td>2.00</td><td>96</td><td></td><td></td><td></td><td></td><td></td>
      <td>L1</td></tr>
    <tr class="bsl"><td></td><td></td><td><sup>3</sup>/<sub>2</sub></td>
      <td>4</td><td>[7.1]</td><td>0.1</td><td></td><td></td><td></td><td></td>
      <td></td><td></td><td></td><td></td><td>L2</td></tr>
    <tr class="bsl"><td>auto</td><td>term</td><td>1</td><td>3</td><td>8.0</td>
      <td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td>
      <td>L3</td></tr>
    </tbody></table></body></html>
    """
    result = parse_levels_html(html, spectrum="Cu I", query_url="https://example.test")
    assert result.database_version == "5.12"
    assert result.reported_level_count == 3
    assert len(result.levels) == 3
    assert result.levels[1].configuration == "3d^104s"
    assert result.levels[1].energy_ev == 7.1
    assert [level.energy_ev for level in result.levels if level.is_bound(7.72638)] == [0.0, 7.1]


def test_parse_nist_level_html_without_lande_column() -> None:
    html = """
    <html><body><p>1 Level Found</p>
    <p>Data on Land&eacute; factors are not available for this ion in ASD</p><table><tbody>
    <tr class="bsl"><td>3p6</td><td>1S</td><td>0</td><td>1</td><td>0</td><td></td>
      <td></td><td>100</td><td></td><td></td><td></td><td></td><td></td><td></td>
      <td>L1</td></tr>
    </tbody></table></body></html>
    """
    result = parse_levels_html(html, spectrum="X I")
    assert len(result.levels) == 1
    assert result.levels[0].lande == ""
    assert result.levels[0].leading_percentages == "100"


def test_parse_ionization_csv() -> None:
    text = '''Sp. Name,Ion Charge,Prefix,Ionization Energy (eV),Suffix,Uncertainty (eV),References,
"=""Cu I""","=""0""","=""""","=""7.726380""","=""""","=""0.000004""","=""L1""",
"=""Cu II""","=""+1""","=""[""","=""20.29239""","=""]""","=""0.1""","=""L2""",
Notes:,,,,,,,
'''
    result = parse_ionization_csv(text, symbol="Cu")
    assert [row.charge for row in result.energies] == [0, 1]
    assert [row.energy_ev for row in result.energies] == [7.72638, 20.29239]
    assert result.energies[1].prefix == "["
    assert result.energies[1].suffix == "]"
