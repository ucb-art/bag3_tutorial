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

from pybag.enum import MinLenMode

from bag.util.math import HalfInt
from bag.util.immutable import Param
from bag.design.module import Module
from bag.layout.template import TemplateDB
from bag.layout.routing.base import TrackID

from xbase.layout.enum import MOSWireType
from xbase.layout.mos.base import MOSBasePlaceInfo, MOSBase

from ..schematic.inv import bag3_tutorial__inv


class InvCore(MOSBase):
    """A single inverter.
    """

    def __init__(self, temp_db: TemplateDB, params: Param, **kwargs: Any) -> None:
        MOSBase.__init__(self, temp_db, params, **kwargs)

    @classmethod
    def get_schematic_class(cls) -> Optional[Type[Module]]:
        return bag3_tutorial__inv

    @classmethod
    def get_params_info(cls) -> Dict[str, str]:
        return dict(
            pinfo='The MOSBasePlaceInfo object.',
            seg='segments of transistors',
            seg_p='segments of pmos',
            seg_n='segments of nmos',
            stack_p='number of transistors in a stack.',
            stack_n='number of transistors in a stack.',
            w_p='pmos width, can be list or integer if all widths are the same.',
            w_n='pmos width, can be list or integer if all widths are the same.',
            ridx_p='pmos row index.',
            ridx_n='nmos row index.',
            sig_locs='Optional dictionary of user defined signal locations',
        )

    @classmethod
    def get_default_param_values(cls) -> Dict[str, Any]:
        return dict(
            seg=-1,
            seg_p=-1,
            seg_n=-1,
            stack_p=1,
            stack_n=1,
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

        seg: int = self.params['seg']
        seg_p: int = self.params['seg_p']
        seg_n: int = self.params['seg_n']
        w_p: int = self.params['w_p']
        w_n: int = self.params['w_n']
        stack_p: int = self.params['stack_p']
        stack_n: int = self.params['stack_n']
        ridx_p: int = self.params['ridx_p']
        ridx_n: int = self.params['ridx_n']
        sig_locs: Mapping[str, Union[float, HalfInt]] = self.params['sig_locs']

        if seg_p <= 0:
            seg_p = seg
        if seg_n <= 0:
            seg_n = seg
        if seg_p <= 0 or seg_n <= 0:
            raise ValueError('Invalid segments.')

        hm_layer = self.conn_layer + 1
        vm_layer = hm_layer + 1
        if self.top_layer < vm_layer:
            raise ValueError(f'MOSBasePlaceInfo top layer must be at least {vm_layer}')

        # --- Step 2: Placement --- #
        nports = self.add_mos(ridx_n, 0, seg_n, w=w_n, stack=stack_n)
        pports = self.add_mos(ridx_p, 0, seg_p, w=w_p, stack=stack_p)

        self.set_mos_size()

        # --- Step 3: Routing --- #
        # Route output
        tr_manager = self.tr_manager
        tr_w_h = tr_manager.get_width(hm_layer, 'sig')
        tr_w_v = tr_manager.get_width(vm_layer, 'sig')
        nout_tidx = sig_locs.get('nout', self.get_track_index(ridx_n, MOSWireType.DS_GATE,
                                                              wire_name='sig', wire_idx=0))
        pout_tidx = sig_locs.get('pout', self.get_track_index(ridx_p, MOSWireType.DS_GATE,
                                                              wire_name='sig', wire_idx=-1))
        nout_tid = TrackID(hm_layer, nout_tidx, tr_w_h)
        pout_tid = TrackID(hm_layer, pout_tidx, tr_w_h)

        pout = self.connect_to_tracks(pports.d, pout_tid, min_len_mode=MinLenMode.NONE)
        nout = self.connect_to_tracks(nports.d, nout_tid, min_len_mode=MinLenMode.NONE)

        vm_tidx = sig_locs.get('out', grid.coord_to_track(vm_layer, pout.middle,
                                                            mode=RoundMode.NEAREST))
        vm_tid = TrackID(vm_layer, vm_tidx, width=tr_w_v)
        self.add_pin('out', self.connect_to_tracks([pout, nout], vm_tid))

        self.add_pin(f'pout', pout, hide=True)
        self.add_pin(f'nout', nout, hide=True)

        # Route input
        in_tidx = sig_locs.get('in', None)
        if in_tidx is None:
            in_tidx = sig_locs.get('nin', None)
            if in_tidx is None:
                default_tidx = self.get_track_index(ridx_n, MOSWireType.G,
                                                    wire_name='sig', wire_idx=0)
                in_tidx = sig_locs.get('pin', default_tidx)

        in_warr = self.connect_to_tracks([nports.g, pports.g],
                                            TrackID(hm_layer, in_tidx, width=tr_w_h))
        self.add_pin('in', in_warr)
        self.add_pin('pin', in_warr, hide=True)
        self.add_pin('nin', in_warr, hide=True)

        # Route supplies
        xr = self.bound_box.xh
        ns_tid = self.get_track_id(ridx_n, False, wire_name='sup')
        ps_tid = self.get_track_id(ridx_p, True, wire_name='sup')
        vss = self.connect_to_tracks(nports.s, ns_tid, track_lower=0, track_upper=xr)
        vdd = self.connect_to_tracks(pports.s, ps_tid, track_lower=0, track_upper=xr)
        self.add_pin('VDD', vdd)
        self.add_pin('VSS', vss)

        # --- Step 4: Schematic parameters --- #
        default_wp = self.place_info.get_row_place_info(ridx_p).row_info.width
        default_wn = self.place_info.get_row_place_info(ridx_n).row_info.width
        thp = self.place_info.get_row_place_info(ridx_p).row_info.threshold
        thn = self.place_info.get_row_place_info(ridx_n).row_info.threshold
        lch = self.place_info.lch
        self.sch_params = dict(
            seg_p=seg_p,
            seg_n=seg_n,
            lch=lch,
            w_p=default_wp if w_p == 0 else w_p,
            w_n=default_wn if w_n == 0 else w_n,
            th_n=thn,
            th_p=thp,
            stack_p=stack_p,
            stack_n=stack_n,
        )


