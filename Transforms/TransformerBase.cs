using MachineLearning.Data;
using Microsoft.ML;

namespace MachineLearning.Transforms
{
   /// <summary>
   /// Classe base per i transformers
   /// </summary>
   public abstract class TransformerBase<T> : DataTransformer<T> where T : class
   {
      #region Fields
      /// <summary>
      /// Transformer
      /// </summary>
      private ITransformer transformer;
      #endregion
      #region Properties
      /// <summary>
      /// Lo schema di input
      /// </summary>
      public abstract DataSchema InputSchema { get; }
      /// <summary>
      /// La pipe di trasformazioni
      /// </summary>
      internal abstract IEstimator<ITransformer> Pipe { get; }
      /// <summary>
      /// Il transformer
      /// </summary>
      public sealed override ITransformer Transformer
      {
         get
         {
            if (transformer != null)
               return transformer;
            var dataView = DataViewGrid.Create(Transforms.GetChannelProvider(), InputSchema);
            return transformer = Pipe.Fit(dataView);
         }
         protected set { }
      }
      /// <summary>
      /// Catalogo trasformazioni
      /// </summary>
      protected TransformsCatalog Transforms { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="transformsCatalog">Catalogo di trasformazini</param>
      internal TransformerBase(TransformsCatalog transformsCatalog) : base(transformsCatalog.GetChannelProvider()) => Transforms = transformsCatalog;
      #endregion
   }
}
