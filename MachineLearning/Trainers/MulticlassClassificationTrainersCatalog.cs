using Microsoft.ML.Trainers;
using Microsoft.ML.Vision;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers multiclasse
   /// </summary>
   public class MulticlassClassificationTrainersCatalog
   {
      #region Fields
      /// <summary>
      /// Contesto di machine learning
      /// </summary>
      private readonly MachineLearningContext _ml;
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="ml"></param>
      internal MulticlassClassificationTrainersCatalog(MachineLearningContext ml) => _ml = ml;
      /// <summary>
      /// Restituisce un trainer di tipo SdcaNonCalibratedMulticlass
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public SdcaNonCalibratedMulticlass SdcaNonCalibrated(SdcaNonCalibratedMulticlassTrainer.Options options = default) => new SdcaNonCalibratedMulticlass(_ml, options);
      /// <summary>
      /// Restituisce un trainer di tipo ImageClassification
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public ImageClassification ImageClassification(ImageClassificationTrainer.Options options = default) => new ImageClassification(_ml, options);
      #endregion
   }
}
