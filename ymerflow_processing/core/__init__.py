"""Method-agnostic processing: anything that serves any airborne survey.

Line selection, the data object, the runners, uncertainty propagation,
decimation, culling policy, and acquisition QC all live here. Nothing in this
package may assume electromagnetic data, gates, moments, or transient decay.

Acquisition QC is the work that proves whether this package is genuinely
method-agnostic — altitude, attitude, speed and line adherence apply to any
airborne survey, so building it here tests the abstraction before the method
packages depend on it.
"""
