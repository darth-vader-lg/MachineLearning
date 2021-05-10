using Microsoft.ML;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearning.Transforms
{
   /// <summary>
   /// Lista di estimatori
   /// </summary>
   /// <typeparam name="TTransformer"></typeparam>
   public class EstimatorList : List<IEstimator<ITransformer>>
   {
      /// <summary>
      /// Costruttore
      /// </summary>
      public EstimatorList() { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="collection">Collezione di estimatori</param>
      public EstimatorList(IEnumerable<IEstimator<ITransformer>> collection) : base(collection) { }
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="capacity">Capacita' iniziale</param>
      public EstimatorList(int capacity) : base(capacity) { }
      /// <summary>
      /// Restituisce la pipe costruita dalla lista di estimatori
      /// </summary>
      /// <returns></returns>
      public IEstimator<ITransformer> GetPipe()
      {
         // Crea la pipe
         var pipe = Count < 1 ? null : Count > 1 ? this[0].Append(this[1]) : this[0];
         for (var i = 2; i < Count; i++)
            pipe = pipe.Append(this[i]);
         return pipe;
      }
   }
}
