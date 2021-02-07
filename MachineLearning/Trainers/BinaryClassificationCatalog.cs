using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di classificazione binaria
   /// </summary>
   public class BinaryClassificationCatalog : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal BinaryClassificationCatalog(IContextProvider<MLContext> context) : base(context) { }
      /// <summary>
      /// Restituisce un trainer di tipo AveragedPerceptron
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public AveragedPerceptronTrainer AveragedPerceptron(Microsoft.ML.Trainers.AveragedPerceptronTrainer.Options options = default) =>
         new AveragedPerceptronTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo AveragedPerceptron
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public FastForestBinaryTrainer FastForest(Microsoft.ML.Trainers.FastTree.FastForestBinaryTrainer.Options options = default) =>
         new FastForestBinaryTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo AveragedPerceptron
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public FastTreeBinaryTrainer FastTree(Microsoft.ML.Trainers.FastTree.FastTreeBinaryTrainer.Options options = default) =>
         new FastTreeBinaryTrainer(this, options);
      /// <summary>
      /// Restituisce un trainer di tipo AveragedPerceptron
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public FieldAwareFactorizationMachineTrainer FieldAwareFactorizationMachine(Microsoft.ML.Trainers.FieldAwareFactorizationMachineTrainer.Options options = default) =>
         new FieldAwareFactorizationMachineTrainer(this, options);
      #endregion
   }
}
