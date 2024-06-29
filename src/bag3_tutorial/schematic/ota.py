# BSD 3-Clause License
#
# Copyright (c) 2018, Regents of the University of California
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

# -*- coding: utf-8 -*-

from typing import Mapping, Any

import pkg_resources
from pathlib import Path

from bag.design.module import Module
from bag.design.database import ModuleDB
from bag.util.immutable import Param


# noinspection PyPep8Naming
class bag3_tutorial__ota(Module):
    """Module for library bag3_tutorial cell ota.

    Fill in high level description here.
    """

    yaml_file = pkg_resources.resource_filename(__name__,
                                                str(Path('netlist_info',
                                                         'ota.yaml')))

    def __init__(self, database: ModuleDB, params: Param, **kwargs: Any) -> None:
        Module.__init__(self, self.yaml_file, database, params, **kwargs)

    @classmethod
    def get_params_info(cls) -> Mapping[str, str]:
        """Returns a dictionary from parameter names to descriptions.

        Returns
        -------
        param_info : Optional[Mapping[str, str]]
            dictionary from parameter names to descriptions.
        """
        return dict(
            lch='channel length in resolution units.',
            w_p='pmos width, in number of fins or resolution units.',
            w_n='nmos width, in number of fins or resolution units.',
            th_p='pmos threshold flavor.',
            th_n='nmos threshold flavor.',
            seg_dp='segments of diff pair',
            seg_load='segments of load',
            seg_tail='segments of tail',
            seg_mirr='segments of mirror',
            out_diff = 'True if differential output. False by default',
            pmos_input='True to use pmos input, nmos load. False by default'
        )

    @classmethod
    def get_default_param_values(cls) -> Mapping[str, Any]:
        return dict(out_diff=False, pmos_input=False)

    def design(self, lch: int, w_p: int, w_n: int, th_p: int, th_n: int,
                seg_dp: int, seg_load: int, seg_tail:int, seg_mirr: int,
                out_diff: bool, pmos_input: bool) -> None:
        """To be overridden by subclasses to design this module.

        This method should fill in values for all parameters in
        self.parameters.  To design instances of this module, you can
        call their design() method or any other ways you coded.

        To modify schematic structure, call:

        rename_pin()
        delete_instance()
        replace_instance_master()
        reconnect_instance_terminal()
        restore_instance()
        array_instance()
        """
        if pmos_input:
            w_dp = w_tail = w_p
            w_load = w_n
            th_dp = th_tail = th_p
            th_load = th_n
            mos_dp = mos_tail = 'pch'
            mos_load = 'nch'
        else:
            w_dp = w_tail = w_n
            w_load = w_p
            th_dp = th_tail = th_n
            th_load = th_p
            mos_dp = mos_tail = 'nch'
            mos_load = 'pch'

        self.design_transistor('XDPL', w=w_dp, lch=lch, seg=seg_dp, intent=th_dp, mos_type=mos_dp)
        self.design_transistor('XDPR', w=w_dp, lch=lch, seg=seg_dp, intent=th_dp, mos_type=mos_dp)

        self.design_transistor('XLOADL', w=w_load, lch=lch, seg=seg_load, intent=th_load, mos_type=mos_load)
        self.design_transistor('XLOADR', w=w_load, lch=lch, seg=seg_load, intent=th_load, mos_type=mos_load)

        self.design_transistor('XMIRR', w=w_tail, lch=lch, seg=seg_mirr, intent=th_tail, mos_type=mos_tail)
        self.design_transistor('XTAIL', w=w_tail, lch=lch, seg=seg_tail, intent=th_tail, mos_type=mos_tail)

        if pmos_input:
            self.reconnect_instance('XDPL', [('B', 'VDD')])
            self.reconnect_instance('XDPR', [('B', 'VDD')])

            self.reconnect_instance('XMIRR', [('S', 'VDD'), ('B', 'VDD')])
            self.reconnect_instance('XTAIL', [('S', 'VDD'), ('B', 'VDD')])

            self.reconnect_instance('XLOADL', [('S', 'VSS'), ('B', 'VSS')])
            self.reconnect_instance('XLOADR', [('S', 'VSS'), ('B', 'VSS')])

        if out_diff:
            self.add_pin('bias_l', 'input')
            self.rename_pin('out', 'outp')
            self.add_pin('outn', 'output')

            self.reconnect_instance_terminal('XLOADL', 'D', 'outn')
            self.reconnect_instance_terminal('XDPL', 'D', 'outn')

            self.reconnect_instance_terminal('XLOADR', 'D', 'outp')
            self.reconnect_instance_terminal('XDPR', 'D', 'outp')
