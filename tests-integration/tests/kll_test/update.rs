// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::panic::AssertUnwindSafe;
use std::panic::catch_unwind;

use datasketches::common::SearchCriteria;
use datasketches::kll::KllSketch;

const DEFAULT_K: u16 = 200;

#[test]
fn zero_weight_is_a_no_op() {
    let mut sketch = KllSketch::<i64>::new(DEFAULT_K).unwrap();
    sketch.update_with_weight(1, 0);
    assert!(sketch.is_empty());
    assert_eq!(sketch.min_item(), None);
}

#[test]
fn weighted_update_matches_repeated_updates_in_exact_mode() {
    let mut weighted = KllSketch::<i64>::new(DEFAULT_K).unwrap();
    let mut repeated = KllSketch::<i64>::new(DEFAULT_K).unwrap();
    for item in 1..=15 {
        weighted.update_with_weight(item, item as u64);
        for _ in 0..item {
            repeated.update(item);
        }
    }
    assert!(!weighted.is_estimation_mode());
    assert_eq!(weighted, repeated);
}

#[test]
fn single_item_with_large_weight() {
    let mut sketch = KllSketch::<i64>::new(DEFAULT_K).unwrap();
    let weight = (1u64 << 40) + 12_345;
    sketch.update_with_weight(7, weight);
    assert_eq!(sketch.n(), weight);
    assert_eq!(sketch.num_retained(), weight.count_ones() as usize);
    assert_eq!(sketch.min_item(), Some(&7));
    assert_eq!(sketch.max_item(), Some(&7));
    assert_eq!(sketch.rank(&7, SearchCriteria::Inclusive).unwrap(), 1.0);
    assert_eq!(sketch.rank(&7, SearchCriteria::Exclusive).unwrap(), 0.0);
    assert_eq!(
        sketch.quantile(0.5, SearchCriteria::Inclusive).unwrap(),
        Some(7)
    );
}

#[test]
fn maximum_weight() {
    let mut sketch = KllSketch::<i64>::new(DEFAULT_K).unwrap();
    sketch.update(1);
    sketch.update_with_weight(2, u64::MAX - 1);
    assert_eq!(sketch.n(), u64::MAX);
    assert_eq!(sketch.min_item(), Some(&1));
    assert_eq!(sketch.max_item(), Some(&2));
    assert_eq!(
        sketch.quantile(0.5, SearchCriteria::Inclusive).unwrap(),
        Some(2)
    );

    let decoded = KllSketch::<i64>::deserialize(&sketch.serialize()).unwrap();
    assert_eq!(decoded, sketch);
}

#[test]
fn weighted_updates_into_a_full_sketch() {
    let mut sketch = KllSketch::<i64>::new(DEFAULT_K).unwrap();
    for item in 0..1_000 {
        sketch.update(item);
    }
    sketch.update_with_weight(-1, 1_000);
    sketch.update_with_weight(2_000, 3_000);
    assert_eq!(sketch.n(), 5_000);
    assert_eq!(sketch.min_item(), Some(&-1));
    assert_eq!(sketch.max_item(), Some(&2_000));
    assert_eq!(sketch.rank(&-1, SearchCriteria::Exclusive), Some(0.0));
    assert_eq!(sketch.rank(&2_000, SearchCriteria::Inclusive), Some(1.0));
    let decoded = KllSketch::<i64>::deserialize(&sketch.serialize()).unwrap();
    assert_eq!(decoded.n(), sketch.n());
}

#[test]
fn weighted_updates_in_estimation_mode() {
    let mut sketch = KllSketch::<i64>::new(DEFAULT_K).unwrap();
    let weights: Vec<u64> = (0..10_000).map(|item| (item % 13) * 1_000 + 1).collect();
    for (item, &weight) in weights.iter().enumerate() {
        sketch.update_with_weight(item as i64, weight);
    }
    let total: u64 = weights.iter().sum();
    assert_eq!(sketch.n(), total);
    assert!(sketch.is_estimation_mode());
    assert_eq!(sketch.min_item(), Some(&0));
    assert_eq!(sketch.max_item(), Some(&9_999));

    let mut previous_rank = 0.0;
    for item in 0..10_000 {
        let rank = sketch.rank(&item, SearchCriteria::Exclusive).unwrap();
        assert!((previous_rank..=1.0).contains(&rank));
        previous_rank = rank;
    }

    let decoded = KllSketch::<i64>::deserialize(&sketch.serialize()).unwrap();
    assert_eq!(decoded.n(), sketch.n());
    assert_eq!(decoded.num_retained(), sketch.num_retained());
}

#[test]
fn weighted_update_of_strings() {
    let mut sketch = KllSketch::<String>::new(DEFAULT_K).unwrap();
    sketch.update_with_weight("b".to_string(), 3);
    sketch.update_with_weight("a".to_string(), 1_000);
    sketch.update_with_weight("c".to_string(), 1 << 20);
    assert_eq!(sketch.n(), 3 + 1_000 + (1 << 20));
    assert_eq!(sketch.min_item().map(String::as_str), Some("a"));
    assert_eq!(sketch.max_item().map(String::as_str), Some("c"));
}

#[test]
fn weight_overflow_preserves_state() {
    let mut sketch = KllSketch::<i64>::new(DEFAULT_K).unwrap();
    sketch.update_with_weight(0, u64::MAX - 1);
    let before = sketch.serialize();

    assert!(catch_unwind(AssertUnwindSafe(|| sketch.update_with_weight(1, 2))).is_err());
    assert!(sketch.serialize() == before, "overflow changed the sketch");

    sketch.update_with_weight(1, 1);
    assert_eq!(sketch.n(), u64::MAX);
    assert_eq!(sketch.max_item(), Some(&1));
}

#[test]
fn weighted_update_at_capacity_boundaries_preserves_weight() {
    for k in [8, 64, 200] {
        for prefix in [0, u64::from(k) - 1, u64::from(k), 2 * u64::from(k)] {
            for weight in [0, 1, u64::from(k) - 1, u64::from(k), u64::from(k) + 1] {
                let mut sketch = KllSketch::new(k).unwrap();
                for _ in 0..prefix {
                    sketch.update(7_i64);
                }
                sketch.update_with_weight(7, weight);
                assert_eq!(sketch.n(), prefix + weight);
                if prefix + weight > 0 {
                    assert_eq!(
                        sketch.quantile(0.5, SearchCriteria::Inclusive).unwrap(),
                        Some(7)
                    );
                    assert_eq!(sketch.rank(&7, SearchCriteria::Inclusive), Some(1.0));
                }
                let decoded = KllSketch::<i64>::deserialize(&sketch.serialize()).unwrap();
                assert_eq!(decoded.n(), prefix + weight);
            }
        }
    }
}

#[test]
fn large_weighted_values_preserve_exact_mass_without_compaction() {
    let mut sketch = KllSketch::<i64>::new(DEFAULT_K).unwrap();
    sketch.update_with_weight(10, 1 << 30);
    sketch.update_with_weight(20, 3 << 30);
    assert_eq!(sketch.n(), 4 << 30);
    assert_eq!(sketch.rank(&10, SearchCriteria::Inclusive), Some(0.25));
    assert_eq!(sketch.rank(&20, SearchCriteria::Exclusive), Some(0.25));
    assert_eq!(
        sketch.quantile(0.2, SearchCriteria::Inclusive).unwrap(),
        Some(10)
    );
    assert_eq!(
        sketch.quantile(0.5, SearchCriteria::Inclusive).unwrap(),
        Some(20)
    );
}

#[test]
fn weighted_updates_preserve_sorted_levels_across_merge_and_roundtrip() {
    for k in [8, 64, 200] {
        let mut sketch = KllSketch::<i64>::new(k).unwrap();
        let mut expected_n = 0;
        for round in 0..8 {
            for index in 0..257 {
                let value = (index * 131 % 257) as i64 - 128;
                let weight = (1u64 << (index % 12)) + (index % 3) as u64;
                sketch.update_with_weight(value, weight);
                expected_n += weight;
            }
            let mut ordinary = KllSketch::<i64>::new(k).unwrap();
            for value in -100..100 {
                ordinary.update(value);
            }
            sketch.merge(&ordinary).unwrap();
            expected_n += 200;
            let decoded = KllSketch::<i64>::deserialize(&sketch.serialize()).unwrap();
            assert_eq!(decoded, sketch, "round {round}, k {k}");
            assert_eq!(decoded.n(), expected_n);
            assert_eq!(decoded.min_item(), Some(&-128));
            assert_eq!(decoded.max_item(), Some(&128));
            let mut previous = 0.0;
            for value in -128..=128 {
                let rank = decoded.rank(&value, SearchCriteria::Inclusive).unwrap();
                assert!(rank >= previous);
                previous = rank;
            }
            assert_eq!(previous, 1.0);
            sketch = decoded;
        }
    }
}
