using MachineLearning.TensorFlow;
using System;

namespace MachineLearning.Model
{
   /// <summary>
   /// Classe base per i modelli TensorFlow
   /// </summary>
   [Serializable]
   public abstract partial class ModelBaseTensorFlow : ModelBase, IContextProvider<TFContext>
   {
      #region Properties
      /// <summary>
      /// Contesto
      /// </summary>
      public TFContext Context => ((IContextProvider<TFContext>)GetChannelProvider()).Context; 
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      public ModelBaseTensorFlow(IContextProvider<TFContext> contextProvider = default) : base(contextProvider ?? MachineLearningContext.Default) { }
      #endregion
   }
}
