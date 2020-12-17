﻿using Microsoft.ML;
using System;

namespace ML.Utilities
{
   /// <summary>
   /// Contesto di machine learning
   /// </summary>
   [Serializable]
   public class MachineLearningContext
   {
      #region Fields
      /// <summary>
      /// Contesto ML.NET
      /// </summary>
      [NonSerialized]
      private MLContext net;
      /// <summary>
      /// Seme per le operazioni random
      /// </summary>
      private readonly int? seed;
      #endregion
      #region Properties
      /// <summary>
      /// Contesto ML.NET
      /// </summary>
      public MLContext NET { get => net ??= new MLContext(seed); private set { net = value; } }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="seed">Seme per le operazioni random</param>
      public MachineLearningContext(int? seed = null) => this.seed = seed;
      #endregion
   }
}
