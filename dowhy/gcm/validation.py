"""Contains a method to reject the causal graph and validate causal causal mechanisms such as post non-linear models.
"""
from enum import Enum, auto
from typing import Any, Callable, Dict, Tuple, Union, Optional, Set

import networkx as nx
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from dowhy.gcm.cms import InvertibleStructuralCausalModel
from dowhy.gcm.graph import DirectedGraph, get_ordered_predecessors, validate_causal_graph, is_root_node
from dowhy.gcm.independence_test import kernel_based


class RejectionResult(Enum):
    REJECTED = auto()
    NOT_REJECTED = auto(),


def reject_causal_structure(causal_graph: DirectedGraph,
                            data: pd.DataFrame,
                            independence_test: Callable[[np.ndarray, np.ndarray], float] = kernel_based,
                            conditional_independence_test: Callable[
                                [np.ndarray, np.ndarray, np.ndarray], float]
                            = kernel_based,
                            significance_level: float = 0.05,
                            fdr_control_method: Optional[str] = 'fdr_bh') \
        -> Tuple[RejectionResult, Dict[str, Dict[str, Dict[str, Union[bool, float, Dict[str, Union[bool, float]]]]]]]:
    """Validates the assumptions in a causal graph against data. To this end, at each node, we test if the node is dependent on each of its parents, and test the local Markov condition.

    :param causal_graph: A directed acyclic graph (DAG).
    :param data: Observations of variables in the DAG.
    :param independence_test: Independence test to use for checking edge dependencies.
    :param conditional_independence_test: Conditional independence test to use for checking local Markov condition.
    :param significance_level: Significance level for (conditional) independence tests.
    :param fdr_control_method: Method for false discovery rate (FDR) control. For various options, please refer to `this page <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html>`_.
    :return: Outcome of the validation process. The first element of the tuple indicates whether the graph is valid w.r.t. given data, and the second element gives the summary of tests at each node. An example for X->Y->Z:

    .. code-block:: python

        [True, {'X': {'local_markov_test': {}, 'edge_dependence_test': {}},
                'Y': {'local_markov_test': {}, 'edge_dependence_test': {'X': {'p_value': 0.5, 'fdr_adjusted_p_value': 0.5, 'success': True}}},
                'Z': {'local_markov_test': {'p_value': 0.0, 'fdr_adjusted_p_value': 0.5, 'success': False},
                      'edge_dependence_test': {'Y': {'p_value': 0.5, 'fdr_adjusted_p_value': 0.5, 'success': True}}}}]
    """
    is_dag_valid = True
    validation_summary = dict()
    all_p_values = []

    for node in causal_graph.nodes:
        parents = get_ordered_predecessors(causal_graph, node)
        non_descendants = _get_non_descendants(causal_graph, node, exclude_parents=True)

        lmc_test_result = dict()
        if parents and non_descendants:
            # test local Markov condition, null=conditional independence
            lmc_p_value = conditional_independence_test(data[node].values,
                                                        data[non_descendants].values,
                                                        data[parents].values)
            lmc_test_result = dict(p_value=lmc_p_value)
            all_p_values.append(lmc_p_value)

        # test edge dependence, null=independence
        edge_dependence_result = dict()
        for parent in parents:
            edge_p_value = independence_test(data[parent].values, data[node].values)
            edge_dependence_result[parent] = dict(p_value=edge_p_value)
            all_p_values.append(edge_p_value)

        validation_summary[node] = dict(
            local_markov_test=lmc_test_result,
            edge_dependence_test=edge_dependence_result
        )

    if fdr_control_method is None:
        successes = np.array(all_p_values) <= significance_level
        adjusted_p_values = np.empty(len(successes))
        adjusted_p_values[:] = np.nan
    else:
        multipletests_result = multipletests(all_p_values, significance_level, method=fdr_control_method)
        successes = multipletests_result[0]
        adjusted_p_values = multipletests_result[1]

    # The order of the p-values added to the list is deterministic.
    index = 0
    for node in causal_graph.nodes:
        if 'p_value' in validation_summary[node]['local_markov_test']:
            validation_summary[node]['local_markov_test']['fdr_adjusted_p_value'] \
                = adjusted_p_values[index]
            validation_summary[node]['local_markov_test']['success'] \
                = not successes[index]
            is_dag_valid &= not successes[index]
            index += 1

        for parent in get_ordered_predecessors(causal_graph, node):
            validation_summary[node]['edge_dependence_test'][parent]['fdr_adjusted_p_value'] \
                = adjusted_p_values[index]
            validation_summary[node]['edge_dependence_test'][parent]['success'] \
                = successes[index]
            is_dag_valid &= successes[index]
            index += 1
    if is_dag_valid:
        return RejectionResult.NOT_REJECTED, validation_summary
    else:
        return RejectionResult.REJECTED, validation_summary


def validate_pnl_assumptions(causal_model: InvertibleStructuralCausalModel,
                             data: pd.DataFrame,
                             independence_test: Callable[[np.ndarray, np.ndarray], float] = kernel_based,
                             significance_level: float = 0.05,
                             fdr_control_method: Optional[str] = None) -> bool:
    """Validate the assumption that the structural causal models can be represented by :py:class:`PostNonlinearSEM <causality.sem.PostNonlinearSEM>` (e.g.
    an :py:class:`AdditiveNoiseModel <causality.sem.AdditiveNoiseModel>`). For this, it is checked if residuals of a model is independent of the model input
    (i.e. we assume causal sufficiency here). This is, :py:class:`PostNonlinearSEM <causality.sem.PostNonlinearSEM>` represent
        Y = f(g(X) + N),
    where f is invertible (g does not need to be), X are the parents of Y and N is (assumed to be) independent noise.
    The latter point is important here. For the given data, this method reconstructs N (i.e. the residual) and performs
    an independence test between X and N.

    Related paper:
        Zhang, K., and A. Hyvärinen.
        On the Identifiability of the Post-Nonlinear Causal Model.
        25th Conference on Uncertainty in Artificial Intelligence (UAI 2009). AUAI Press, 2009.

    :param causal_model: A fitted invertible structural causal model.
    :param data: Observations of variables in the DAG.
    :param independence_test: Independence test to use for checking if residual and input are dependent.
    :param significance_level: Significance level for deciding whether input and residual is dependent.
    :param fdr_control_method: Method for false discovery rate (FDR) control. For various options, please refer to `this page <https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html>`_.
    :return: True if the all causal models are consistent with the post non-linear model assumption.
    """
    validate_causal_graph(causal_model.graph)

    all_p_values = []

    for node in causal_model.graph.nodes:
        if is_root_node(causal_model.graph, node):
            continue
        else:
            parents_samples = data[get_ordered_predecessors(causal_model.graph, node)].to_numpy()
            residuals = causal_model.causal_mechanism(node).estimate_noise(data[node].to_numpy(), parents_samples)

            all_p_values.append(independence_test(parents_samples, residuals))

    if fdr_control_method is None or len(all_p_values) == 0:
        return np.all(np.array(all_p_values) > significance_level)
    else:
        return np.all(multipletests(all_p_values, significance_level, method=fdr_control_method)[0])


def _get_non_descendants(causal_graph: DirectedGraph, node: Any, exclude_parents: bool = False) -> Set[Any]:
    nodes_to_exclude = nx.descendants(causal_graph, node).union({node})
    if exclude_parents:
        nodes_to_exclude = nodes_to_exclude.union(causal_graph.predecessors(node))
    return set(causal_graph.nodes).difference(nodes_to_exclude)
