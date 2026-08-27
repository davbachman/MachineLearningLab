import { describe, expect, it } from 'vitest'
import { codeLabQuestions, knnEvaluationNotebookLab } from '../data/codeLabQuestions'
import { getCorrectCodeLabSubmission } from '../lib/codeLab'

describe('Code Lab question specifications', () => {
  it('provides the same three autogradable stages for every existing topic', () => {
    expect(Object.keys(codeLabQuestions)).toEqual([
      'numpy',
      'pca',
      'kmeans',
      'knn',
      'decisionTrees',
      'randomForests',
    ])

    for (const question of Object.values(codeLabQuestions)) {
      expect(question.stages.map((stage) => stage.kind)).toEqual([
        'executionTrace',
        'diagnoseMutation',
        'distinguishingTest',
      ])
      expect(question.stages.map((stage) => stage.id)).toEqual(['TRACE', 'MUTATION', 'TEST'])
      const expectedVariants: Record<string, string> = {
        'numpy-notebook-lab': 'numpy-notebook-v1',
        'pca-code-lab': 'pca-notebook-v1',
        'kmeans-code-lab': 'kmeans-code-v4',
        'knn-code-lab': 'knn-notebook-v1',
        'decision-tree-code-lab': 'decision-tree-notebook-v1',
        'rf-code-lab': 'rf-notebook-v1',
      }
      expect(question.variantId).toBe(expectedVariants[question.id])
    }
  })

  it('provides an explicit executable setup and invocation for every trace', () => {
    const expectedInvocations = {
      numpy: 'array_shape = Y.shape',
      pca: 'projected = pca.fit_transform(X_trace)',
      kmeans: 'labels, updated_centroids = lloyd_step(X, centroids)',
      knn: 'prediction = knn.predict(3.0, 1.5)',
      decisionTrees: 'best_split = BestSplit(X_trace, y_trace)',
      randomForests: 'forest.fit(X_trace, y_trace)',
    }

    for (const [topic, question] of Object.entries(codeLabQuestions)) {
      expect(question.fixtureTitle.trim().length).toBeGreaterThan(0)
      expect(question.fixture.trim().length).toBeGreaterThan(0)
      expect(question.invocationTitle.trim().length).toBeGreaterThan(0)
      expect(question.invocation).toContain(
        expectedInvocations[topic as keyof typeof expectedInvocations],
      )
      expect(question.stages.every((stage) => stage.prompt.trim().length > 0)).toBe(true)
    }
  })

  it('accepts a complete ID-based answer and rejects a changed trace answer', () => {
    for (const question of Object.values(codeLabQuestions)) {
      const correct = getCorrectCodeLabSubmission(question)
      expect(question.validator(correct).correct).toBe(true)

      const changed = structuredClone(correct)
      const firstField = question.stages[0].fields[0]
      changed.stages.TRACE[firstField.id] = firstField.options.find(
        (option) => option.id !== firstField.correctOptionId,
      )!.id

      expect(question.validator(changed).correct).toBe(false)
    }
  })

  it('uses unique fixed IDs within every lab', () => {
    for (const question of Object.values(codeLabQuestions)) {
      const fieldIds = question.stages.flatMap((stage) =>
        stage.fields.map((field) => `${stage.id}.${field.id}`),
      )
      expect(new Set(fieldIds).size).toBe(fieldIds.length)

      for (const stage of question.stages) {
        for (const field of stage.fields) {
          const optionIds = field.options.map((option) => option.id)
          expect(new Set(optionIds).size).toBe(optionIds.length)
          expect(optionIds).toContain(field.correctOptionId)
        }
      }
    }
  })

  it('keeps machine IDs separate from student-facing labels', () => {
    for (const question of Object.values(codeLabQuestions)) {
      for (const stage of question.stages) {
        for (const field of stage.fields) {
          expect(field.label).not.toBe(field.id)
          for (const option of field.options) {
            expect(option.label).not.toBe(option.id)
            expect(option.label).not.toMatch(/^(M_|TEST_)/)
          }
        }
      }
    }
  })

  it('uses concise expression-first labels for every trace field', () => {
    expect(codeLabQuestions.numpy.stages[0].fields.map((field) => field.label)).toEqual([
      'array_shape',
      'column_shape',
      'slice_value',
      'filtered_values',
      'summary',
      'matrix_summary',
    ])
    expect(codeLabQuestions.pca.stages[0].fields.map((field) => field.label)).toEqual([
      'pca.basis.shape',
      'projected.tolist()',
      'new_projected.tolist()',
    ])
    expect(codeLabQuestions.kmeans.stages[0].fields.map((field) => field.label)).toEqual([
      'labels.tolist()',
      'updated_centroids.tolist()',
      'post_objective',
    ])
    expect(codeLabQuestions.knn.stages[0].fields.map((field) => field.label)).toEqual([
      'X.shape after X = X[:, 2:] in the notebook',
      'columns remaining in X after the slice',
      'distances.tolist()',
      'neighbor_species.tolist()',
      'prediction',
    ])
    expect(codeLabQuestions.randomForests.stages[0].fields.map((field) => field.label)).toEqual([
      'samples.tolist() when i == 0',
      'features.tolist() when i == 0',
      'samples.tolist() when i == 1',
      'features.tolist() when i == 1',
      'prediction',
    ])
    expect(codeLabQuestions.decisionTrees.stages[0].fields.map((field) => field.label)).toEqual([
      'impurity',
      'mask.tolist()',
      'split_impurity',
      'best_split',
      'left_prediction, right_prediction',
    ])

    for (const question of Object.values(codeLabQuestions)) {
      expect(question.stages[1].fields).toHaveLength(1)
      expect(question.stages[2].fields.map((field) => field.label)).toEqual(['Test'])
    }
  })

  it('shows code-shaped values without invented assignments or mappings', () => {
    const correctLabel = (topic: keyof typeof codeLabQuestions, fieldId: string) => {
      const field = codeLabQuestions[topic].stages[0].fields.find(
        (candidate) => candidate.id === fieldId,
      )!
      return field.options.find((option) => option.id === field.correctOptionId)!.label
    }

    expect(correctLabel('numpy', 'ARRAY_SHAPE')).toBe('(100, 10)')
    expect(correctLabel('numpy', 'COLUMN_SHAPE')).toBe('(1000, 1)')
    expect(correctLabel('numpy', 'SLICE_VALUE')).toBe('36')
    expect(correctLabel('numpy', 'SUMMARY')).toBe('(8.483314773547882, 996, 555)')
    expect(correctLabel('numpy', 'MATRIX_SUMMARY')).toBe('((100, 100), 555.0)')
    expect(correctLabel('pca', 'BASIS_SHAPE')).toBe('(2, 1)')
    expect(correctLabel('pca', 'PROJECTED')).toBe('[[-2.0], [0.0], [2.0]]')
    expect(correctLabel('pca', 'NEW_PROJECTED')).toBe('[[-1.0], [1.0]]')
    expect(correctLabel('kmeans', 'ASSIGNMENTS')).toBe('[0, 0, 1, 1]')
    expect(correctLabel('kmeans', 'POST_OBJECTIVE')).toBe('4.0')
    expect(correctLabel('knn', 'IRIS_X_SHAPE')).toBe('(150, 2)')
    expect(correctLabel('knn', 'DISTANCES')).toBe('[4.25, 1.25, 3.25, 10.25]')
    expect(correctLabel('knn', 'NEIGHBOR_SPECIES')).toBe('[0, 0]')
    expect(correctLabel('knn', 'PREDICTION')).toBe('0')
    expect(correctLabel('randomForests', 'SAMPLES_0')).toBe('[4, 0, 3, 3, 3]')
    expect(correctLabel('randomForests', 'FEATURES_0')).toBe('[0, 2]')
    expect(correctLabel('randomForests', 'SAMPLES_1')).toBe('[3, 4, 0, 1, 3]')
    expect(correctLabel('randomForests', 'FEATURES_1')).toBe('[3, 2]')
    expect(correctLabel('randomForests', 'PREDICTION')).toBe('1')
    expect(correctLabel('decisionTrees', 'IMPURITY')).toBe('0.5')
    expect(correctLabel('decisionTrees', 'MASK')).toBe('[False, False, True, True]')
    expect(correctLabel('decisionTrees', 'SPLIT_IMPURITY')).toBe('0.0')
    expect(correctLabel('decisionTrees', 'BEST_SPLIT')).toBe('(0, 2.0)')
    expect(correctLabel('decisionTrees', 'PREDICTIONS')).toBe('0, 1')
  })

  it('uses uniquely diagnostic and executable mutation tests', () => {
    expect(codeLabQuestions.numpy.code).toContain('P = np.matmul(X, Y.T)')
    expect(codeLabQuestions.numpy.stages[1].prompt).toContain('P = np.matmul(X.T, Y)')
    const numpyTest = codeLabQuestions.numpy.stages[2].fields[0]
    const correctNumpyTest = numpyTest.options.find(
      (option) => option.id === numpyTest.correctOptionId,
    )!
    expect(correctNumpyTest.description).toContain('P.shape == (100, 100)')
    expect(correctNumpyTest.description).toContain('P[5, 5] == s[5]')

    expect(codeLabQuestions.knn.code).toContain(
      'indices = distances.argsort()[:self.n_neighbors]',
    )
    expect(codeLabQuestions.knn.stages[1].prompt).toContain(
      'distances.argsort()[-self.n_neighbors:]',
    )

    const knnTest = codeLabQuestions.knn.stages[2].fields[0]
    const correctKnnTest = knnTest.options.find(
      (option) => option.id === knnTest.correctOptionId,
    )!
    expect(correctKnnTest.description).toContain(
      'SpeciesOfNeighbors(3.0, 1.5).tolist() == [0, 0]',
    )

    const forestTest = codeLabQuestions.randomForests.stages[2].fields[0]
    const correctForestTest = forestTest.options.find(
      (option) => option.id === forestTest.correctOptionId,
    )!
    expect(correctForestTest.description).toContain('voting_forest.predict')
    expect(correctForestTest.description).toContain('== 1')

    const kmeansTest = codeLabQuestions.kmeans.stages[2].fields[0]
    const correctKmeansTest = kmeansTest.options.find(
      (option) => option.id === kmeansTest.correctOptionId,
    )!
    expect(correctKmeansTest.description).toContain('lloyd_step(X, centroids)')
    expect(correctKmeansTest.description).toContain('assert np.allclose')

    const treeTest = codeLabQuestions.decisionTrees.stages[2].fields[0]
    const correctTreeTest = treeTest.options.find(
      (option) => option.id === treeTest.correctOptionId,
    )!
    expect(correctTreeTest.description).toContain('root.predict(np.array([2.0])) == 1')
  })

  it('models the second k-NN notebook as a separate autogradable lab', () => {
    expect(knnEvaluationNotebookLab.variantId).toBe('knn-evaluation-notebook-v1')
    expect(knnEvaluationNotebookLab.stages.map((stage) => stage.kind)).toEqual([
      'executionTrace',
      'diagnoseMutation',
      'distinguishingTest',
    ])
    expect(knnEvaluationNotebookLab.fixture).toContain(
      'test_indices = np.random.choice(np.arange(size), test_size)',
    )

    const correct = getCorrectCodeLabSubmission(knnEvaluationNotebookLab)
    expect(knnEvaluationNotebookLab.validator(correct).correct).toBe(true)

    const trace = knnEvaluationNotebookLab.stages[0]
    const correctLabel = (fieldId: string) => {
      const field = trace.fields.find((candidate) => candidate.id === fieldId)!
      return field.options.find((option) => option.id === field.correctOptionId)!.label
    }
    expect(correctLabel('ORIGINAL_X_SHAPE')).toBe('(150, 4)')
    expect(correctLabel('REQUESTED_TEST_SIZE')).toBe('30')
    expect(correctLabel('ACTUAL_TEST_SIZE')).toBe('27')
    expect(correctLabel('ACTUAL_TRAIN_SIZE')).toBe('123')
    expect(correctLabel('TEST_DATA_SHAPE')).toBe('(27, 4)')

    const samplingChange = knnEvaluationNotebookLab.stages[1].fields[0]
    expect(
      samplingChange.options.find(
        (option) => option.id === samplingChange.correctOptionId,
      )!.label,
    ).toBe('30 test rows and 120 training rows')

    const modelSelection = knnEvaluationNotebookLab.stages[2].fields[0]
    expect(
      modelSelection.options.find(
        (option) => option.id === modelSelection.correctOptionId,
      )!.description,
    ).toContain('k[accuracies.argmax()] == 5')
  })
})
