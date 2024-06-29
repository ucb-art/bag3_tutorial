# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2018 Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Any, Dict, Optional, Union, Mapping, Type, List

from pybag.enum import MinLenMode, RoundMode

from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.design.module import Module
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from ..schematic.ota import bag3_tutorial__ota


class OTACore(MOSBase):
    """A 5T OTA with tail mirror
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_tutorial__ota

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg_dp='segments for diff pair',
            seg_load='segments for load',
            seg_tail='segments for tail',
            seg_mirr='segments for mirr',
            out_diff = 'True if differential output'
            # w_p='pmos width, can be list or integer if all widths are the same.',
            # w_n='pmos width, can be list or integer if all widths are the same.',
            # ridx_p='pmos row index.',
            # ridx_n='nmos row index.',
            # sig_locs='Optional dictionary of user defined signal locations',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            out_diff=True,
            w_p=0,
            w_n=0,
            ridx_p=-1,
            ridx_n=0,
            sig_locs={},
        )

    def draw_layout(self) -> None:
        # --- Step 1: Process parameters and draw base --- #
        pinfo = MOSBasePlaceInfo.make_place_info(self.grid, self.params['pinfo'])
        self.draw_base(pinfo)

        grid = self.grid

        seg_dp: int = self.params['seg_dp']
        seg_load: int = self.params['seg_load']
        seg_tail: int = self.params['seg_tail']
        seg_mirr: int = self.params['seg_mirr']
        out_diff: bool = self.params['out_diff']

        assert seg_mirr % 2 == 0, "Feature to be added"
        assert seg_tail % 2 == 0, "Feature to be added"
        
        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        # --- Step 2: Floorplan and Placement --- #

        # Floorplan
        seg0 = seg_tail + seg_mirr
        if seg0 % 2: 
            seg0 += 1  # Enforce even
        seg1 = seg_dp * 2
        seg2 = seg_load * 2

        seg_tot = max([seg0, seg1, seg2])
        seg_2 = seg_tot // 2

        ridx0, ridx1, ridx2 = 0, 1, 2

        # Placement
        mirr_l = self.add_mos(ridx0, seg_2 - seg_mirr // 2, seg_mirr // 2)
        mirr_r = self.add_mos(ridx0, seg_2 + seg_mirr // 2, seg_mirr // 2, flip_lr=True)

        tail_l = self.add_mos(ridx0, seg_2 - seg_mirr // 2 , seg_tail // 2,  flip_lr=True)
        tail_r = self.add_mos(ridx0, seg_2 + seg_mirr // 2 , seg_tail // 2)

        dp_p = self.add_mos(ridx1, seg_2, seg_dp, flip_lr=True)
        dp_n = self.add_mos(ridx1, seg_2, seg_dp)

        load_l = self.add_mos(ridx2, seg_2, seg_load, flip_lr=True)
        load_r = self.add_mos(ridx2, seg_2, seg_load)

        self.set_mos_size()

        # --- Step 3: Routing --- #
        tr_manager = self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')

        # Route bias
        _mirr = [mirr_l.d, mirr_r.d, mirr_l.g, mirr_r.g]
        self.connect_wires(_mirr)
        _g = [mirr_l.g, mirr_r.g, tail_l.g, tail_r.g]
        bias_tidx = self.get_track_index(ridx0, MOSWireType.G, wire_name='sig', wire_idx=0)
        bias_tid = TrackID(hm_layer, bias_tidx, tr_w_h)
        bias = self.connect_to_tracks(_g, bias_tid, min_len_mode=MinLenMode.NONE)
        self.add_pin('bias', bias)

        # Route tail  
        tail0_tidx = self.get_track_index(ridx0, MOSWireType.DS, wire_name='sig', wire_idx=0)
        tail0_tid = TrackID(hm_layer, tail0_tidx, tr_w_h)
        tail0 = self.connect_to_tracks([tail_l.d, tail_r.d], tail0_tid, min_len_mode=MinLenMode.NONE)

        tail1_tidx = self.get_track_index(ridx1, MOSWireType.DS, wire_name='sig', wire_idx=0)
        tail1_tid = TrackID(hm_layer, tail1_tidx, tr_w_h)
        tail1 = self.connect_to_tracks([dp_p.s, dp_n.s], tail1_tid, min_len_mode=MinLenMode.NONE)

        xm = self.bound_box.xm
        tail_tidx = self.grid.coord_to_track(vm_layer, xm, mode=RoundMode.NEAREST)
        tail_tid_vm = TrackID(vm_layer, tail_tidx, tr_w_v)
        tail = self.connect_to_tracks([tail0, tail1], tail_tid_vm)
        self.add_pin('tail', tail, hide=True)

        # Route inputs
        inp_tidx = self.get_track_index(ridx1, MOSWireType.G, wire_name='sig', wire_idx=0)
        inn_tidx = self.get_track_index(ridx1, MOSWireType.G, wire_name='sig', wire_idx=1)

        inp, inn = self.connect_differential_tracks(dp_p.g, dp_n.g, hm_layer, inp_tidx, inn_tidx, width=tr_w_h)
        self.add_pin('inp', inp)
        self.add_pin('inn', inn)

        # Route outputs
        out1_tidx = self.get_track_index(ridx1, MOSWireType.DS, wire_name='sig', wire_idx=-1)
        out1_tid = TrackID(hm_layer, out1_tidx, tr_w_h)
        out1l_hm = self.connect_to_tracks(dp_p.d, out1_tid)
        out1r_hm = self.connect_to_tracks(dp_n.d, out1_tid)

        out2_tidx = self.get_track_index(ridx2, MOSWireType.DS, wire_name='sig', wire_idx=0)
        out2_tid = TrackID(hm_layer, out2_tidx, tr_w_h)
        out2l_hm = self.connect_to_tracks(load_l.d, out2_tid)
        out2r_hm = self.connect_to_tracks(load_r.d, out2_tid)

        _, _tmp = self.tr_manager.place_wires(vm_layer, ['sig', 'sig', 'sig'], center_coord=xm)
        _tmp = [TrackID(vm_layer, _tidx, tr_w_v) for _tidx in _tmp]
        outl_vm_tid, _, outr_vm_tid = _tmp
        out_l = self.connect_to_tracks([out1l_hm, out2l_hm], outl_vm_tid)
        out_r = self.connect_to_tracks([out1r_hm, out2r_hm], outr_vm_tid)

        # Route load gates
        load_tidx = self.get_track_index(ridx2, MOSWireType.G, wire_name='sig', wire_idx=0)
        load_tid = TrackID(hm_layer, load_tidx, tr_w_h)
        load_hm = self.connect_to_tracks([load_l.g, load_r.g], load_tid)
        if out_diff:
            # Add bias node
            self.add_pin('bias_l', load_hm)
            self.add_pin('outn', out_l)
            self.add_pin('outp', out_r)
        else:
            # Diode connect
            self.connect_to_track_wires(load_hm, load_l.d)
            self.add_pin('out', out_r)

        # Route supplies
        xr = self.bound_box.xh
        ns_tid = self.get_track_id(ridx0, False, wire_name='sup')
        ps_tid = self.get_track_id(ridx2, True, wire_name='sup')
        vss = self.connect_to_tracks([mirr_l.s, mirr_r.s, tail_l.s, tail_r.s], ns_tid, track_lower=0, track_upper=xr)
        vdd = self.connect_to_tracks([load_l.s, load_r.s], ps_tid, track_lower=0, track_upper=xr)
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)

        # --- Step 4: Schematic parameters --- #
        default_wp = self.place_info.get_row_place_info(ridx2).row_info.width
        default_wn = self.place_info.get_row_place_info(ridx0).row_info.width
        thp = self.place_info.get_row_place_info(ridx2).row_info.threshold
        thn = self.place_info.get_row_place_info(ridx0).row_info.threshold
        lch = self.place_info.lch
        self.sch_params = dict(
            lch=lch,
            w_p=default_wp,
            w_n=default_wn,
            th_n=thn,
            th_p=thp,
            seg_dp=seg_dp,
            seg_load=seg_load,
            seg_tail=seg_tail,
            seg_mirr=seg_mirr,
            out_diff=out_diff
        )


